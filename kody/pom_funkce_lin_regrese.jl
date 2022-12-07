function P( w, b, X)
    return X*w .+ b
end

function grad_w(X,y,w,b)
    return transpose(X)*(X*w .+ b .- y), sqrt(sum((X*w .+ b - y).*(X*w .+ b - y)))
end

function grad_b(X,y,w,b)
    g = 0
    for i=1:size(X)[1]
        g = g + sum(X[i,:].*w) + b -y[i]
    end
    return g
end

function grad_descend(X, y, w0, b0, step_w,step_b,max_iter, resid_crit, vypis)
    i = 0
    g_norm_w = 1.0
    g_norm_b = 1.0

    g_init, loss = grad_w(X,y,w0,b0)
    loss_prev = 2*loss
    resid = 1

    while (i<max_iter &&  resid > resid_crit)

        i=i+1
        g_w, loss = grad_w(X,y,w0,b0)
        g_b = grad_b(X,y,w0,b0)

        w0 = w0 .- step_w.*g_w
        b0 = b0 - step_b*g_b

        g_norm_w = sqrt(sum(g_w.*g_w))
        g_norm_b = abs(g_b)

        resid = (loss_prev-loss)/loss
        loss_prev = loss

        if mod(i, vypis)==0
            println("  iteration = ", i)
            println("    current w = ", w0, ", b = ", b0)
            println("    loss function = ", loss)
            println("    ||grad_w|| = ", g_norm_w)
            println("    ||grad_b|| = ", g_norm_b)
            println("    resid. = ", resid)
            println()
        end
    end

    return w0, b0, g_norm_w, g_norm_b, loss, i
end




function plot_lin_regrese_results(param, X, y, w, header, output_folder)

        title_string = header[end]*" = "
        for i = 1:size(X)[2]
                title_string = title_string*header[i]*"("*string(round(w[i],digits=3))*")"
                if i<size(X)[2]
                        title_string = title_string*" + "
                end
        end

    max, i_max = findmax(abs.(w))
    x_viz = zeros(2,length(param))
    for j = 1:length(param)
            if j==i_max
                    x_viz[1,j] = minimum(X[:,j])
                    x_viz[2,j] = maximum(X[:,j])
            else
                    x_viz[1,j] = mean(X[:,j])
                    x_viz[2,j] = mean(X[:,j])
            end
    end


    fig = figure(figsize=( 6,4)  , dpi=300)
    plot(x_viz[:,i_max], P(w, x_viz), "m--", label = "Fitted model")
    scatter(X[:,i_max], y, s=10, label = "Observations")
    legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=2.0)
    xlabel(header[param[i_max]])
    ylabel(header[end])
    title(title_string)
    savefig(output_folder*"fit"*string(param)*".png", bbox_inches="tight")
    close()

    open(output_folder * "w_"*string(param)*".txt", "w") do io
            for i = 1:length(param)
                write(io, header[param[i]], " : ")
                writedlm(io, w[i])
            end
    end;

end

function plot_lin_regrese_wBias_results(param, X, y, w, b, header, output_folder)

        title_string = header[end]*" = "*string(round(b,digits=3))* " + "
        for i = 1:size(X)[2]
                title_string = title_string*header[i]*"("*string(round(w[i],digits=3))*")"
                if i<size(X)[2]
                        title_string = title_string*" + "
                end
        end

    max, i_max = findmax(abs.(w))
    x_viz = zeros(2,length(param))
    for j = 1:length(param)
            if j==i_max
                    x_viz[1,j] = minimum(X[:,j])
                    x_viz[2,j] = maximum(X[:,j])
            else
                    x_viz[1,j] = mean(X[:,j])
                    x_viz[2,j] = mean(X[:,j])
            end
    end


    fig = figure(figsize=( 6,4)  , dpi=300)
    plot(x_viz[:,i_max], P(w, b, x_viz), "m--", label = "Fitted model")
    scatter(X[:,i_max], y, s=10, label = "Observations")
    legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=2.0)
    xlabel(header[param[i_max]])
    ylabel(header[end])
    title(title_string)
    savefig(output_folder*"fit"*string(param)*".png", bbox_inches="tight")
    close()

    open(output_folder * "w.txt", "w") do io
            write(io, "Bias = ")
            writedlm(io, b)
            for i = 1:size(X)[2]
                write(io, header[i]* " : ")
                writedlm(io, w[i])
            end
    end;
end


function export_histograms(header, data, output_folder)

        for i =1:size(data)[2]-1
                fig = figure(figsize=( 6,4)  , dpi=300)
                hist(data[:,i])
                title(header[i])
                savefig(output_folder*"hist_X"*string(i)*".png", bbox_inches="tight")
                close()
        end

        i = size(data)[2]

        fig = figure(figsize=( 6,4)  , dpi=300)
        hist(data[:,i])
        title(header[i])
        savefig(output_folder*"hist_y.png", bbox_inches="tight")
        close()
end

function plot_zavislosti(header, data, output_folder)

        for i =1:size(data)[2]-1
                fig = figure(figsize=( 6,4)  , dpi=300)
                scatter(data[:,i], data[:,end])
                xlabel(header[i])
                ylabel(header[end])
                savefig(output_folder*"zavislost_y_X"*string(i)*".png", bbox_inches="tight")
                close()
        end
end
