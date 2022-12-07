function generate_data_log_regrese()
    mean1 = [2.,3.]
    C = [1.5 0; 0 1.8]
    d = MvNormal(mean1, C)
    x1 = rand(d, 200)'

    mean2 = [-4.5,2.75]
    C2 = [6.5 0; 0 8.1]
    d2 = MvNormal(mean2, C2)
    x2 = rand(d2, 250)'

    mean3 = [6.25,6.]
    C3 = [1.5 0; 0 2.8]
    d3 = MvNormal(mean3, C3)
    x3 = rand(d3, 250)'

    mean4 = [2.5,-2.75]
    C4 = [14.5 0; 0 3.1]
    d4 = MvNormal(mean4, C4)
    x4 = rand(d4, 350)'

    y1 = -1.0.*ones(size(x1)[1])
    y2 = ones(size(x2)[1])
    y3 = -1.0.*ones(size(x3)[1])
    y4 = 1.0.*ones(size(x4)[1])


    x1 = vcat(x1,x3)
    y1 = vcat(y1,y3)

    x2 = vcat(x2,x4)
    y2 = vcat(y2,y4)

    X = vcat(x1,x2)
    Y = vcat(y1,y2)

    return x1, y1, x2, y2
end


function plot_init_data(x1, y1, x2, y2, folder)

        l1 = "Class " *string(y1[1])
        l2 = "Class " *string(y2[1])

        fig = figure(figsize=( 6,4)  , dpi=300)
        scatter(x1[:, 1], x1[:,2], c="m", s=3, label = l1)
        scatter(x2[:, 1], x2[:,2],  c="b", s=3, label = l2)
        legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=2.0)
        savefig(folder*"vstupni_data.png", bbox_inches="tight")
        close()
end

function log_r_grad_w(X,y,w,b)

    m = 1.0./(1.0 .+ exp.(y.*((X*w) .+ b[1])) )
    g = (-m.*y).*X
    return sum(g,dims=1)'
end

function log_r_grad_b(X,y,w,b)

    m = 1.0./(1.0 .+ exp.(y.*((X*w) .+ b[1])) )
    g = -(m.*y)

    return sum(g,dims=1)'
end

function log_grad_descend(X,y,w0,b0, step_w, step_b, l1, l2)
    i = 0
    g_norm_w = 1.0
    g_norm_b = 1.0

    while (i<5.0e6 && (g_norm_w > 0.025 || g_norm_b > 0.025))

        i=i+1
        g_w = grad_w(X,y,w0,b0) .+ l1.*w0
        g_b = grad_b(X,y,w0,b0) .+ l2.*b0


        w1 = w0 .- step_w.*(g_w)
        b1 = b0 .- step_b.*(g_b)

        w0 = w1
        b0 = b1

        g_norm_w = sqrt(sum(g_w.*g_w))
        g_norm_b = sqrt(g_b[1]*g_b[1])


        if mod(i, 1.0e4)==0
            println("iter = ", i)
            println("  w = ", w0, ",  b = ", b0[1])
            println("g norm w = ", g_norm_w)
            println("g norm b = ", g_norm_b)
            println()
        end
    end

    return w0, b0
end
