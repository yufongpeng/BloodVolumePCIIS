using BloodVolumePCIIS, Test

function test_show(x)
    try 
        show(IOBuffer(), x)
        true
    catch
        false
    end
end

const file_directory = joinpath(@__DIR__, "data")	# default: current directory
const calibration_keywords = nothing # default: ["volume", "Volume"]
const sample_keywords = [r"B\d+"] # default: ["sample", "Sample"] 

@testset "BloodVolumePCIIS.jl" begin
    files = isnothing(file_directory) ? readdir(".") : joinpath.(Ref(file_directory), readdir(file_directory))
	cal = isnothing(calibration_keywords) ? read_calibration(files) : read_calibration(files; keywords = calibration_keywords)
	sample = isnothing(sample_keywords) ? read_sample(files) : read_sample(files; keywords = sample_keywords)
    @test length(cal) == 26
    @test length(sample) == 2
    @test first(cal.data[1].t) > 0.3
    @test last(cal.data[end].t) < 0.5
    @test extrema(cal.x) == (5, 40)
    model = optimize(cal)
    @test test_show(model)
    et = error_table(model)
    @test extrema(et.volume) == (5, 40)
    @test isapprox(model_volume(model), transform_volume(model).(volume(model)))
    @test isapprox(model_response(model), transform_response(model).(response(model)))
    @test isapprox(err(model) ./ volume(model), pcterr(model))
    cit = volume(model, cal)
    @test isapprox(unique(average_sample(cit).volume), et.predicted_volume)
end