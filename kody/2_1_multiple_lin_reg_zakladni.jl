using PyPlot
using Revise
using Distributions
using DelimitedFiles
using CSV
using DataFrames

include("pom_funkce_lin_regrese.jl")

data_folder = "../data/"

df = readdlm(data_folder*"Real_estate_1.csv",  ',', '\n')
header = string.(df[1,:])
data = Float64.(df[2:end,:])
println(size(data))
println("Parameters : ", header)


output_base = "../vysledky/"
if !isdir(output_base)
        mkdir(output_base)
end

output_folder = output_base * "multiple_linear_regrese_zakladni/"
if !isdir(output_folder)
        mkdir(output_folder)
end

################################################################################

# vykresleni uvodnich statistik

export_histograms(header, data, output_folder)
plot_zavislosti(header, data, output_folder)

################################################################################
# param = sloupce matice X pro linearni regresi
# w = vysledne vahy pro jednotlive sloupce
# vizualizace vysledku :
#        osa x ... sloupec matice X s nejvyssi vahou
#        osa y ... y (cena nemovitosti)

param = [1,2,3] # [1,2,3], [1,2]

X = data[:,param]
y = data[:,end]

w = inv(transpose(X)*X)*transpose(X)*y

println(header[end]," : ")
for i = 1:length(param)
        println("   ", header[param[i]] ," ... w = ", w[i] )
end
println()


plot_lin_regrese_results(param, X, y, w, header, output_folder)
