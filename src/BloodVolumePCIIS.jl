module BloodVolumePCIIS
using CSV, TypedTables, DataPipes, GLM, MLStyle, SplitApplyCombine, Statistics, IterTools, InverseFunctions
using InverseFunctions: square
import GLM: formula, coef
import Base: show
export Model, 
        parse_cal_name, parse_sample_name, 
        read_pciis, read_calibration, read_sample,
        integrate, optimize, inverse_predict, 
        predictor, response, model_predictor, model_response, transform_predictor, transform_response,
        pcterr, err, error_table,
        volume, average_sample,

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

Additional information about names is processed by `parse_fn`.
"""
read_calibration(files; keywords = ["volume", "Volume"], parse_fn = parse_cal_name) =
    read_files(files, keywords, parse_fn)

"""
    read_sample(files; keywords = ["sample", "Sample"], parse_fn = parse_sample_name)

Read multiple raw chromatograms files created by Agilent MassHunter software which contain the `keywords`.

Additional information about names is processed by `parse_fn`.
"""
read_sample(files; keywords = ["sample", "Sample"], parse_fn = parse_sample_name) = 
    read_files(files, keywords, parse_fn)
    
"""
    read_files(files, keywords, parse_fn)

Read multiple raw chromatograms files created by Agilent MassHunter software which contain the `keywords`.

Additional information about names is processed by `parse_fn`.
"""
function read_files(files, keywords, parse_fn)
    files = filter(x -> (endswith(x, ".csv") || endswith(x, ".CSV")) && any(occursin(k, x) for k in keywords), files)
    tbl = @p files map(parse_fn) map(Table) append!(__...)
    data = @p files read_pciis map(filter!(x -> 0.3 < x[2] < 0.5, _))
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
function optimize(tbl; left_max_points = 5, right_max_points = 5, fs = [@formula(inv(y) ~ x), @formula((inv ∘ square)(y) ~ x)])
    gtbl = @p tbl groupview(getproperty(:id))
    id = @p gtbl map(sum(x -> x.y, _.data)) map(findmin(_)[2])
    rt = @p zip(id, gtbl) mean(mean(x -> x.t[_[1]], _[2].data))
    models = map(Iterators.product(fs, 0:left_max_points, 1:right_max_points)) do (f, wl, wr)
        dt = integrate(tbl, rt, wl, wr)
        dt, lm(f, dt)
    end
    Δ = @p tbl.data map(partition(_.t, 2, 1)) map(map(x -> foldl(-, x), _)) mean(mean)
    sspe, id = findmin(models) do (dt, model)
        x = @view model.mm.m[:, 2]
        y = model.model.rr.y
        x1, i1 = findmin(x)
        x2, i2 = findmax(x)
        sum((_inverse_predict(model, y, (y[i1] - y[i2]) / (x1 - x2) > 0) ./ dt.x .- 1) .^ 2)
    end
    lb = Tuple(id)[2] - 1
    rb = Tuple(id)[3]
    x = @view models[id][2].mm.m[:, 2]
    y = models[id][2].model.rr.y
    x1, i1 = findmin(x)
    x2, i2 = findmax(x)
    Model(models[id][2], rt, (lb, rb), (y[i1] - y[i2]) / (x1 - x2) > 0, integrate(tbl, rt, lb, rb))
end

predictor(model::Model) = model.data.x
response(model::Model) = model.data.y
model_predictor(model) = model.model.pp.X[:, 2]
model_response(model) = model.model.rr.y
transform_predictor(model) = get_fn(formula(model).rhs.terms[2])
transform_response(model) = get_fn(formula(model).lhs)
coef(model) = model.model.pp.beta0

for fn in (:model_predictor, :model_response, :transform_predictor, :transform_response, :coef)
    @eval begin
        $fn(model::Model) = $fn(model.model)
    end
end

inverse_predict(model::Model, y) = inverse_predict(model.model, y, model.corr)
function inverse_predict(model, y, corr)
    β = coef(model)
    fn_y = transform_response(model)
    fn_x = inverse(transform_predictor(model))
    length(β) == 2 && (return fn_x.((fn_y.(y) .- β[1]) ./ β[2]))
    m = corr ? 1 : -1
    map(fn_y(y)) do y
        fn_x((-β[2] .+ sqrt(max(β[2]^2 - 4 * β[3] * (β[1] - y), 0)) .* m) ./ 2 ./ β[3])
    end
end

_inverse_predict(model::Model, y = model_response(model)) = _inverse_predict(model.model, y, model.corr)
function _inverse_predict(model, y, corr)
    β = coef(model)
    fn_x = inverse(transform_predictor(model))
    length(β) == 2 && (return fn_x.((y .- β[1]) ./ β[2]))
    m = corr ? 1 : -1
    map(y) do y
        fn_x((-β[2] .+ sqrt(max(β[2]^2 - 4 * β[3] * (β[1] - y), 0)) .* m) ./ 2 ./ β[3])
    end
end

err(model::Model) = _inverse_predict(model) .- predictor(model)
pcterr(model::Model) = _inverse_predict(model) ./ predictor(model) .- 1

function error_table(model::Model)
    X = predictor(model)
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

function volume(model::Model, tbl)
    dt = integrate(tbl, model.rt, model.id...)
    Table(dt, volume = inverse_predict(model, dt.y), data = nothing)
end

get_fn(term::FunctionTerm) = term.forig
get_fn(term) = identity

integrate(model::Model, tbl) = integrate(tbl, model.rt, model.id...)
function integrate(tbl::Table, rt::Float64, left_points::Int, right_points::Int)
    #Δ = @p tbl.data map(partition(_.t, 2, 1)) map(map(foldl(-, _))) map(mean) mean
    id = @p tbl.data map(findmin(x -> abs(x - rt), _.t)[2]) map((_ - left_points):(_ + right_points))
    y = @p zip(id, tbl.data) map(sum(_[2].y[_[1]]) - _[2].y[first(_[1])] / 2 - _[2].y[last(_[1])] / 2) 
    Table(tbl; y)
end

function average_sample(tbl::Table) 
    vol = @p tbl groupview(getproperty(:id)) map(mean(_.volume))
    Table(tbl; volume = get.(Ref(vol), tbl.id, NaN))
end

end