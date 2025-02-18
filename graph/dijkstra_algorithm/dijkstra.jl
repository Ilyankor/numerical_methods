# graph adjacency matrix
adj_matrix = [
    0 5 3 0 0;
    5 0 7 9 0;
    3 7 0 0 3;
    0 9 0 0 4;
    0 0 3 4 0;
]

# initializations
num_vertices = 5
vertices_list = [1, 2, 3, 4, 5] # list of all vertices
start_node = 2 # start node
S = Int64[] # vertices visited
L = [Vector{Float64}(undef,2) for _ in 1:6] # label matrix: [preceding vertex, total weight]

# find the neighbors function
function neighbors(vertices)
    N = Int64[]

    for n in vertices
        for j = 1:num_vertices
            if adj_matrix[n,j] != 0
                push!(N, j)
            else
                continue
            end
        end
    end

    return N
end

# Dijkstra's algorithm
push!(S, start_node)
L[start_node] = [start_node, 0]

for v in vertices_list
    if v == start_node
        continue
    else
        L[v] = [NaN, Inf]
    end
end

while (S != vertices_list)
    
    # vertices and their neighbors
    R = Int64[]
    append!(R, S)
    append!(R, neighbors(S))
    R = unique(R)

    # compute potential next weights
    for r in R
        if r in S
            continue
        else
            for x in neighbors([r])
                if !(x in S)
                    continue
                else
                    if (L[x][2] + adj_matrix[x,r] < L[r][2])
                        L[r] = [x, L[x][2] + adj_matrix[x,r]]
                    end
                end
            end
        end
    end

    # choose the next vertex
    try
        T = [] # temporary storage
        for v in vertices_list
            if v in S
                continue
            else
                push!(T, [v, L[v][2]])
            end
        end
        T_M = mapreduce(permutedims, vcat, T)
        T_index = argmin(T_M[:,2])
        push!(S, T_M[T_index, 1])
    catch MethodError
        break
    end
end

# sum the lengths
len = []
for i = 1:num_vertices
    push!(len, adj_matrix[i, Int64(L[i][1])])
end
total = sum(len)
println(total)