using DataFrames
using MLJ
using CSV
using GZip

load_df() = GZip.open("data.csv.gz") |> CSV.read

function test_lasso(df, the_model)
    y, X = unpack(df, ==(:y), !=(:y))

    the_machine = machine(the_model, X, y)
    train, test = partition(eachindex(y), 0.7, shuffle=false)

    fit!(the_machine, rows=train)

    ŷ = MLJ.predict(the_machine, rows=test)

    unique(ŷ)
end

@load LassoRegressor pkg="MLJLinearModels" name=LassoRegressorMLJ
@load LassoRegressor pkg="ScikitLearn" name=LassoRegressorSKLearn 

let 
    df = load_df()

    if test_lasso(df, LassoRegressorMLJ()) == [0.0] 
        println("LassoRegressor from MLJLinearModels returns the null vector...")
    end

    if test_lasso(df, LassoRegressorSKLearn()) != [0.0]
        println("whereas the LassoRegressor from ScikitLearn returns a vector != 0")
    end
end