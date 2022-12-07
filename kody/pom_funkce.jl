function P( w, x)
    res = zeros(size(x))

    for i=1:size(x)[1]
        for j=1:size(w)[1]
            res[i] = res[i] + w[j]*x[i]^(j-1)
        end
    end
    return res
end


function grad(x,y,w,d)
    g = zeros(d)

    dif = P(w,x) .- y

    for j=1:d
        for i =1:size(x)[1]
            g[j] = g[j] + dif[i]*x[i]^(j-1)
        end
    end

    return 2.0.*g, sqrt(sum(dif.*dif))
end

function grad_descend(x, y, w0, d, step, resid_crit, iter_max, vypis, l)
    i = 0
    g_norm = 1.0

    g_init, loss = grad(x,y,w0,d)
    loss_prev = 2*loss
    resid = 1

    while (i<iter_max && resid > resid_crit )

        i=i+1
        g, loss = grad(x,y,w0,d)
        w0 = w0 - step.*(g + l.*w0)

        g_norm = sqrt(sum(g.*g))
        resid = (loss_prev-loss)/loss
        loss_prev = loss

        if mod(i, vypis)==0
            println("  iteration = ", i)
            println("    current w = ", w0)
            println("    loss function = ", loss)
            println("    ||grad|| = ", g_norm)
            println("    resid. = ", resid)
            println()
        end
    end

    return w0, g_norm, loss, i
end



function plot_init_regrese(a,b,x,y_e,d,output_folder)
    figure(figsize=( 6,4)  , dpi=300)
    pom = collect(a:0.001:b)
    title("Training data")
    plot(pom, sin.(pom), "r--", label ="sin(x)")
    scatter(x,y_e, s=10, label = "Training data (y + e)")
    legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=2.0)
    savefig(output_folder*"training_data_for_degree="*string(d)*".png", bbox_inches="tight")
    close()
end

function plot_results_regrese(a, b, x_train, y_train, w_init, w_final, grad_final, i_final, loss_final, l, d, output_folder)
    fig = figure(figsize=( 6,4)  , dpi=300)

    scatter(x_train, y_train, s=10, label = "Training values")

    plot(collect(a:0.01:b), P(w_init, collect(a:0.01:b)), "g--", label = "Initial estimation")
    plot(collect(a:0.01:b), P(w_final, collect(a:0.01:b)), "m:", label = "Final estimation")
    plot(collect(a:0.01:b), sin.(collect(a:0.01:b)), "r:", label ="sin(x)")

    if l == 0.0
        title("Degree = "*string(d))
    else
        title("Degree = "*string(d)*", regul. = "*string(l))
    end
    legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=2.0)
    savefig(output_folder*"fit_degree="*string(d)*"_regul="*string(l)*".png", bbox_inches="tight")

    close()


    open(output_folder * "final_w_degree="*string(d)*"_regul="*string(l)*".txt", "w") do io
                writedlm(io, w_final)
    end;

    open(output_folder * "convergence_results_degree="*string(d)*"_regul="*string(l)*".txt", "w") do io
        write(io, "||grad|| = ")
        writedlm(io, grad_final)
        write(io, "iterations = ")
        writedlm(io, i_final)
        write(io, "loss = ")
        writedlm(io, loss_final)
    end;
end
