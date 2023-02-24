module BloodVolumePCIIS
using CSV, TypedTables, DataPipes, GLM, MLStyle, SplitApplyCombine, Statistics, IterTools, InverseFunctions
using InverseFunctions: square
import GLM: formula, coef
import Base: show
export Model, 
        parse_cal_name, parse_sample_name, 
        read_pciis, read_calibration, read_sample,
        integrate, optimize, inverse_predict, 
        volume, response, model_volume, model_response, transform_volume, transform_response,
        pcterr, err, error_table, average_sample,

        @formula, lm, Table, inverse, square, predict, formula, coef

# ===========================================================
# Parse
"""
    parse_cal_name(name)

Parse the names of calibration data files in the following format:
DATE_KEYWORD_LEVEL_VOLUME_ANYOTHERTHING.CSV
"""
function parse_cal_name(name)
    s = split(name, "_")
    i = parse(Int, s[3])
    v = parse(Int, s[4])
    (id = [i], x = [v])
end

"""
    parse_sample_name(name)

Parse the names of sample files in the following format:
DATE_KEYWORDwithID-REPEAT.CSV
"""
function parse_sample_name(name)
    s = split(name, "_")
    id, r = split(s[2], "-")
    r = replace(r, "r" => "", ".CSV" => "", ".csv" => "")
    (id = [id], repeat = [parse(Int, r)])
end
# ================================================================
# Input
"""
    read_pciis(files header = [:id, :t, :y])

Read raw chromatograms files created by Agilent MassHunter software.
"""
read_pciis(files; header = [:id, :t, :y]) = 
    CSV.read.(files, Table; skipto = 3, header)

"""
    read_calibration(files; keywords = ["volume", "Volume"], parse_fn = parse_cal_name)

Read multiple raw chromatograms files created by Agilent MassHunter software which contain the `keywords`.

Additional information about names is processed by `parse_fn` which should return a `NamedTuple`: `(id = [...], repeat = [...] )`.
"""
read_calibration(files; keywords = ["volume", "Volume"], parse_fn = parse_cal_name) =
    read_files(files, keywords, parse_fn)

"""
    read_sample(files; keywords = ["sample", "Sample"], parse_fn = parse_sample_name)

Read multiple raw chromatograms files created by Agilent MassHunter software which contain the `keywords`.

Additional information about names is processed by `parse_fn` which should return a `NamedTuple`: `(id = [...], repeat = [...] )`.
"""
read_sample(files; keywords = ["sample", "Sample"], parse_fn = parse_sample_name) = 
    read_files(files, keywords, parse_fn)
    
"""
    read_files(files, keywords, parse_fn)

Read multiple raw chromatograms files created by Agilent MassHunter software which contain the `keywords`.

Additional information about names is processed by `parse_fn` which should return a `NamedTuple`: `(id = [...], repeat = [...] )`.
"""
function read_files(files, keywords, parse_fn)
    id = findall(x -> (endswith(x, ".csv") || endswith(x, ".CSV")) && any(occursin(k, x) for k in keywords), last.(split.(files, "\\")))
    isempty(id) && throw(ArgumentError("No qualified files"))
    fs = @view files[id]
    tbl = @p fs map(parse_fn) map(Table) append!(__...)
    data = @p fs read_pciis map(filter!(x -> 0.3 < x[2] < 0.5, _))
    tbl = Table(tbl; data)
end
# ==================================================================================
# Modeling
struct Model
    model
    rt
    id::Tuple{Int, Int}
    corr::Bool
    data
end

function Base.show(io::IO, model::Model)
    println(io, "∘ Model: ")
    println(io, model.model)
    Δ = @p model.data.data map(partition(_.t, 2, 1)) map(map(x -> foldl(-, x), _)) mean(mean)
    println(io, "∘ r²: ", round(r²(model.model); digits = 4))
    println(io, "∘ SSPE: ", round(sum(x -> x ^ 2, pcterr(model)); digits = 4))
    println(io, "∘ RT: ", round(model.rt + Δ * (model.id[1] + 0.5); digits = 4), " ~ ", round(model.rt - Δ * (model.id[2] + 0.5); digits = 4), " min")
    println(io)
end

"""
    optimize(tbl; left_max_points = 5, right_max_points = 5, fs = [@formula(inv(y) ~ x), @formula((inv ∘ square)(y) ~ x)])

Optimize the integration intervals and model formula based on sum of squared percentage error (SSPE) by grid searching.

The function used in `fs` needs to be invertable, i.e. `inverse(fn)` exists.
"""
function optimize(tbl::Table; left_max_points = 5, right_max_points = 5, fs = [@formula(inv(y) ~ x), @formula((inv ∘ square)(y) ~ x)])
    gtbl = @p tbl groupview(getproperty(:id))
    id_min = @p gtbl map(sum(x -> x.y, _.data)) map(findmin(sum, collect(partition(_, 2, 1)))[2])
    Δ = @p tbl.data map(partition(_.t, 2, 1)) map(map(x -> foldl(-, x), _)) mean(mean)
    rt = @p zip(id_min, gtbl) mean(mean(x -> x.t[_[1]], _[2].data)) +(0.5Δ)
    models = map(Iterators.product(fs, 0:left_max_points, 0:right_max_points)) do (f, wl, wr)
        dt = integrate(tbl, rt, wl, wr)
        dt, lm(f, dt)
    end
    sspe, id = findmin(models) do (dt, model)
        x = @view model.mm.m[:, 2]
        y = model.model.rr.y
        x1, i1 = findmin(x)
        x2, i2 = findmax(x)
        sum((_inverse_predict(model, y, (y[i1] - y[i2]) / (x1 - x2) > 0) ./ dt.x .- 1) .^ 2)
    end
    lb = Tuple(id)[2] - 1
    rb = Tuple(id)[3] - 1
    x = @view models[id][2].mm.m[:, 2]
    y = models[id][2].model.rr.y
    x1, i1 = findmin(x)
    x2, i2 = findmax(x)
    Model(models[id][2], rt, (lb, rb), (y[i1] - y[i2]) / (x1 - x2) > 0, integrate(tbl, rt, lb, rb))
end

"""
    volume(model::Model)

Get volumes utilized in `model`.
"""
volume(model::Model) = model.data.x
"""
    response(model::Model)

Get repsonses utilized in `model`.
"""
response(model::Model) = model.data.y
"""
    model_volume(model)

Get transformed volume utilized in `model`. For instance, if the formula is `@formula(inv(y) ~ 1 + inv(x))`, this function returns `inv(x)`.
"""
model_volume(model) = model.model.pp.X[:, 2]
"""
    model_response(model)

Get transformed repsonses utilized in `model`. For instance, if the formula is `@formula(inv(y) ~ 1 + inv(x))`, this function returns `inv(y)`.
"""
model_response(model) = model.model.rr.y
"""
    transform_volume(model)

Get the function for transforming volume utilized in `model`. For instance, if the formula is `@formula(inv(y) ~ 1 + inv(x))`, this function returns `inv`.
"""
transform_volume(model) = get_fn(formula(model).rhs.terms[2])
"""
    transform_response(model)

Get the function for transforming volume utilized in `model`. For instance, if the formula is `@formula(inv(y) ~ 1 + inv(x))`, this function returns `inv`.
"""
transform_response(model) = get_fn(formula(model).lhs)
"""
    coef(model)

Coefficients of the model.
"""
coef(model) = model.model.pp.beta0

for fn in (:model_volume, :model_response, :transform_volume, :transform_response, :coef)
    @eval begin
        $fn(model::Model) = $fn(model.model)
    end
end

get_fn(term::FunctionTerm) = term.forig
get_fn(term) = identity
# =================================================================================
# Data processing
"""
    integrate(model::Model, tbl::Table)
    integrate(tbl::Table, rt::Float64, left_points::Int, right_points::Int)

Integrate the chromatograms based on fitted `model` or `rt`, `left_points` and `right_points`.
"""
integrate(model::Model, tbl::Table) = integrate(tbl, model.rt, model.id...)
function integrate(tbl::Table, rt::Float64, left_points::Int, right_points::Int)
    #Δ = @p tbl.data map(partition(_.t, 2, 1)) map(map(foldl(-, _))) map(mean) mean
    id = @p tbl.data map(findmin(x -> max(rt - x, 0), _.t)[2]) map((_ - left_points - 1):(_ + right_points))
    y = @p zip(id, tbl.data) map(sum(_[2].y[_[1]]) - _[2].y[first(_[1])] / 2 - _[2].y[last(_[1])] / 2) 
    Table(tbl; y)
end

"""
    volume(model::Model, tbl::Table)

Predicting volumes of samples in `tbl` by `model`.
"""
function volume(model::Model, tbl::Table)
    dt = integrate(tbl, model.rt, model.id...)
    Table(dt, volume = inverse_predict(model, dt.y), data = nothing)
end

"""
    inverse_predict(model::Model, y)
    inverse_predict(model, y, corr)

Predicting volumes for given `y` based on `model::Model` or a native regression model and `corr` determining whether the larger solution is used when there are multiple solution.
"""
inverse_predict(model::Model, y) = inverse_predict(model.model, y, model.corr)
function inverse_predict(model, y, corr)
    β = coef(model)
    fn_y = transform_response(model)
    fn_x = inverse(transform_volume(model))
    length(β) == 2 && (return fn_x.((fn_y.(y) .- β[1]) ./ β[2]))
    m = corr ? 1 : -1
    map(fn_y(y)) do y
        fn_x((-β[2] .+ sqrt(max(β[2]^2 - 4 * β[3] * (β[1] - y), 0)) .* m) ./ 2 ./ β[3])
    end
end

_inverse_predict(model::Model, y = model_response(model)) = _inverse_predict(model.model, y, model.corr)
function _inverse_predict(model, y, corr)
    β = coef(model)
    fn_x = inverse(transform_volume(model))
    length(β) == 2 && (return fn_x.((y .- β[1]) ./ β[2]))
    m = corr ? 1 : -1
    map(y) do y
        fn_x((-β[2] .+ sqrt(max(β[2]^2 - 4 * β[3] * (β[1] - y), 0)) .* m) ./ 2 ./ β[3])
    end
end

"""
    average_sample(tbl::Table) 

Computing avearge volumes per `id`.
"""
function average_sample(tbl::Table) 
    vol = @p tbl groupview(getproperty(:id)) map(mean(_.volume))
    Table(tbl; volume = get.(Ref(vol), tbl.id, NaN))
end

# ==============================================================================
# Performance evaluation
"""
    err(model::Model)

Error or deviation of volumes of calibration points in `model`.
"""
err(model::Model) = _inverse_predict(model) .- volume(model)
"""
    pcterr(model::Model)

Pertentage error or deviation of volumes of calibration points in `model`.
"""
pcterr(model::Model) = _inverse_predict(model) ./ volume(model) .- 1
"""
error_table(model::Model)

Pressent several performance evaluation of `model` including error, percentage error, mean absolute percentage error(MAPE), maximum absolute oercentage error(MaxAPE) and relative standard deviation(RSD). 
"""
function error_table(model::Model)
    X = volume(model)
    pred = _inverse_predict(model)
    err = pred .- X
    pcterr = pred ./ X .- 1
    gtbl = @p Table(volume = X, 
            predicted_volume = pred,
            error = err, 
            percentage_error = pcterr,
            MAPE = abs.(pcterr)) |> 
            group(getproperty(:volume)) 
    tbl = @p gtbl |> 
            map(map(y -> mean(y), columns(_))) |> 
            collect |> Table
    Table(tbl, MaxAPE = (@p gtbl map(maximum(_.MAPE)) collect), RSD = (@p gtbl map(std(_.percentage_error)) collect))
end

end