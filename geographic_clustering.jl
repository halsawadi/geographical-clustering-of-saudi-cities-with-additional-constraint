### A Pluto.jl notebook ###
# v0.17.0

using Markdown
using InteractiveUtils

# ╔═╡ 2d6d0588-f4e4-4cad-8353-9c0ba020fec6
begin
	using FileIO
	using ImageShow
	using ImageIO
	using PlutoUI
	using Cbc
	using DataFrames
	using Distances
	using JuMP
	using LinearAlgebra
end

# ╔═╡ d44f3f70-705b-43a2-85c0-a441e08fbbe6
load("Pydata_Jeddah_22_new-06.png")

# ╔═╡ a12d1135-ce89-414c-866e-14b4415e1990
md"""
# Geographical Clustering of Saudi Cities With Additional Constraint
**Originally Contributed by**: Matthew Helm ([with help from Mathieu Tanneau on Julia Discourse](https://discourse.julialang.org/t/which-jump-jl-solver-for-this-problem/43350/17?u=mthelm85))
**Customized by**: Hassan Alsawadi
"""

# ╔═╡ d35c109a-e5b6-4c01-a9c2-333b883783f2
md"""
#### Intializing packages

_When running this notebook for the first time, this could take up to 5 minutes. Hang in there!_
"""

# ╔═╡ 2877f488-1ab3-46da-8a49-7db65c3d8a45
md"""
## Objective
The goal of this exercise is to cluster 𝑛 cities into 𝑘 groups, minimizing the total pairwise distance between cities and ensuring that the variance in the total populations of each group is relatively small.

For this example, we'll use the 20 most populous cities in Saudi Arabia.
"""

# ╔═╡ cf6d3f77-781d-4cd0-ad45-9c5c58380354
cities = DataFrame(
    city=["Riyadh", "Jeddah", "Mecca", "Medina", "Dammam", "Abha", "Ha'il", "Hofuf", "Al-Mubarraz", "Sakakah", "Taif", "Khamis Mushait", "Jubail", "Buraydah", "Tabuk", "Najran", "Qatif", "Khobar", "Al Bahah", "Hafr Al-Batin"],
    population=[7676654, 4697000, 2042000, 1488782, 1252523, 1093705, 936465, 858395, 837000, 779480, 695000, 667000, 684531, 669000, 667000, 505652, 474573, 455541, 366000, 359000],
    lat=[24.633333, 21.543333, 21.4225, 24.466667, 26.433333, 18.216944, 27.516667, 25.383333, 25.383333, 29.969722, 21.275094, 18.3, 27, 26.333333, 28.397222, 17.491667, 26.556, 26.283333, 20.0125, 28.434167],
    lon=[46.716667, 39.172778, 39.823333, 39.6, 50.1, 42.505278, 41.683333, 49.583333, 49.6, 40.2, 40.406156, 42.733333, 49.666667, 43.966667, 36.578889, 44.132222, 49.996, 50.2, 41.465278, 45.963611])

# ╔═╡ 11c8323a-9cd5-4580-93d9-ebcd085b4551
md"""
## Model Specifics
We will cluster these 20 cities into 3 different groups and we will assume that the ideal or target population  𝑃  for a group is simply the total population of the 20 cities divided by 3:
"""

# ╔═╡ 0a9212cd-4c79-4752-930b-45dce6a1f3e8
begin
	n = size(cities,1)
	k = 3
	P = sum(cities.population) / k
end

# ╔═╡ 7f7439d3-3d3b-4098-8be5-ca9aa59373ce
md"""### Obtaining the distances between each city
Let's leverage the *Distances.jl* package to compute the pairwise Haversine distance between each of the cities in our data
set and store the result in a variable we'll call `dm`:
"""

# ╔═╡ 3e708ffa-05d9-4cf7-ba4c-11ca5a3e43e7
dm = Distances.pairwise(Haversine(6372.8), Matrix(cities[:, [3,4]])', dims=2)

# ╔═╡ 54082ac4-c31d-4d27-9cc3-5248018c6838
md"""Our distance matrix is symmetric so we'll convert it to a `LowerTriangular` matrix so that we can better interpret the
objective value of our model (if we don't do this the total distance will be doubled):"""

# ╔═╡ 6bc42cc7-e3d4-4470-bece-6af950280372
dm2 = LowerTriangular(dm)

# ╔═╡ a5f3abe7-a843-4ea5-8e77-dfd075b243a5
md"""
## Build the model
Now that we have the basics taken  care of, we can set up our model, create decision variables, add constraints, and then
solve.

First, we'll set up a model that leverages the [Cbc](https://github.com/coin-or/Cbc) solver. Next, we'll set up a binary
variable $x_{i,k}$ that takes the value $1$ if city $i$ is in group $k$ and $0$ otherwise. Each city must be in a group, so
we'll add the constraint $\sum_kx_{i,k} = 1$ for every $i$.
"""

# ╔═╡ ee41a6b3-fb13-4269-ab22-ad2a85bf26e5
begin
	model = Model(Cbc.Optimizer)
	
	@variable(model, x[1:n, 1:k], Bin)
	
	for i in 1:n
	    @constraint(model, sum(x[i,:]) == 1)
	end
end

# ╔═╡ dd262e8b-4697-4aa7-9bc7-a32ac25c96dc
md"""
The total population of a group $k$ is $Q_k = \sum_ix_{i,k}q_i$ where $q_i$ is simply the $i$th value from the `population`
column in our `cities` DataFrame. Let's add constraints so that $\alpha \leq (Q_k - P) \leq \beta$. We'll set $\alpha$
equal to -2,500,000 and $\beta$ equal to 2,500,000. By adjusting these thresholds you'll find that there is a tradeoff
between having relatively even populations between groups and having geographically close cities within each group. In
other words, the larger the absolute values of $\alpha$ and $\beta$, the closer together the cities in a group will be but
the variance between the group populations will be higher.
"""

# ╔═╡ d5fc7ca7-33cd-4f8a-aecf-8897bcd40e15
begin
	α = -2_500_000
	β = 2_500_000
	for i in 1:k
	    @constraint(model, (x' * cities.population)[i] - P <= β)
	    @constraint(model, (x' * cities.population)[i] - P >= α)
	end
end

# ╔═╡ 4ab84877-f5d8-4c1b-b0d2-346f0048bb20
md"""
Now we need to add one last binary variable $z_{i,j}$ to our model that we'll use to compute the total distance between the 
cities in our groups, defined as  $\sum_{i,j}d_{i,j}z_{i,j}$. Variable $z_{i,j}$ will equal $1$ if cities $i$ and $j$ are 
in the same group, and $0$ if they are not in the same group.

To ensure that $z_{i,j} = 1$ if and only if cities $i$ and $j$ are in the same group, we add the constraints $z_{i,j} \geq 
x_{i,k} + x_{j,k} - 1$ for every pair $i,j$ and every $k$:
"""

# ╔═╡ 1054b6a4-1c28-46b6-88a4-6f7cae84edee
begin
	@variable(model, z[1:n,1:n], Bin)
	
	for k in 1:k, i in 1:n, j in 1:n
	    @constraint(model, z[i,j] >= x[i,k] + x[j,k] - 1)
	end
end

# ╔═╡ 07f58656-4826-4ce5-8c97-bf5570f0fc60
md"""
We can now add an objective to our model which will simply be to minimize the dot product of $z$ and our distance matrix,
`dm`. We can then call `optimize!` and review the results.
"""

# ╔═╡ 17088a0b-a568-431c-8d01-13f957e2a4c0
begin
	@objective(model, Min, dot(z,dm));
	
	optimize!(model)
end

# ╔═╡ 4ded521e-c41f-4927-854b-86f8912269db
md"""## Reviewing the Results
Now that we have results, we can add a column to our `cities` DataFrame for the group and then loop through our $x$
variable to assign each city to its group. Once we have that, we can look at the total population for each group and also
plot the cities and their groups to verify visually that they are grouped by geographic proximity.
"""

# ╔═╡ dab4e559-9dc6-4a72-8d38-9a0487b6251a
PlutoUI.with_terminal() do
	cities.group = zeros(n)
	
	for i in 1:n, j in 1:k
	    if round(value.(x)[i,j]) == 1.0
	        cities.group[i] = j
	    end
	end
	
	for group in groupby(cities, :group)
	    @show group
	    println("")
	    @show sum(group.population)
	    println("")
	end
end

# ╔═╡ aa340dc7-8833-4e40-b63e-c602096e40e3
md"""
The populations of each group are fairly even and we can see from the plot below that the groupings look good in terms of
geographic proximity:
"""

# ╔═╡ a2fad4bc-66fa-4466-ae29-b495590f035f
load("output.png")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Cbc = "9961bab8-2fa3-5c5a-9d89-47fab24efd76"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
ImageIO = "82e4d734-157c-48bb-816b-45c225c6df19"
ImageShow = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
Cbc = "~0.9.1"
DataFrames = "~1.3.2"
Distances = "~0.10.7"
FileIO = "~1.13.0"
ImageIO = "~0.6.1"
ImageShow = "~0.3.3"
JuMP = "~0.22.3"
PlutoUI = "~0.7.34"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[ASL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6252039f98492252f9e47c312c8ffda0e3b9e78d"
uuid = "ae81ac8f-d209-56e5-92de-9978fef736f9"
version = "0.1.3+0"

[[AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "6f1d9bc1c08f9f4a8fa92e3ea3cb50153a1b40d4"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.1.0"

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "4c10eee4af024676200bc7752e536f858c6b8f93"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.1"

[[BinaryProvider]]
deps = ["Libdl", "Logging", "SHA"]
git-tree-sha1 = "ecdec412a9abc8db54c0efc5548c64dfce072058"
uuid = "b99e7846-7c00-51b0-8f62-c81ae34c0232"
version = "0.5.10"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[Cbc]]
deps = ["BinaryProvider", "CEnum", "Cbc_jll", "Libdl", "MathOptInterface", "SparseArrays"]
git-tree-sha1 = "6656166f484075dd146c9f452b1428116eaf76d4"
uuid = "9961bab8-2fa3-5c5a-9d89-47fab24efd76"
version = "0.9.1"

[[Cbc_jll]]
deps = ["ASL_jll", "Artifacts", "Cgl_jll", "Clp_jll", "CoinUtils_jll", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS32_jll", "Osi_jll", "Pkg"]
git-tree-sha1 = "a3c5986d7713bce4260d9826deead060a17c8e2d"
uuid = "38041ee0-ae04-5750-a4d2-bb4d0d83d27d"
version = "200.1000.501+0"

[[Cgl_jll]]
deps = ["Artifacts", "Clp_jll", "CoinUtils_jll", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Osi_jll", "Pkg"]
git-tree-sha1 = "11eb7b7688925e9751b5d7a187aaa4291eae2664"
uuid = "3830e938-1dd0-5f3e-8b8e-b3ee43226782"
version = "0.6000.300+0"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "7dd38532a1115a215de51775f9891f0f3e1bac6a"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.12.1"

[[ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[Clp_jll]]
deps = ["Artifacts", "CoinUtils_jll", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "METIS_jll", "MUMPS_seq_jll", "OpenBLAS32_jll", "Osi_jll", "Pkg"]
git-tree-sha1 = "b1031dcfbb44553194c9e650feb5ab65e372504f"
uuid = "06985876-5285-5a41-9fcb-8948a742cc53"
version = "100.1700.601+0"

[[CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "2e62a725210ce3c3c2e1a3080190e7ca491f18d7"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.7.2"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[CoinUtils_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS32_jll", "Pkg"]
git-tree-sha1 = "44173e61256f32918c6c132fc41f772bab1fb6d1"
uuid = "be027038-0da8-5614-b30d-e42594cb92df"
version = "200.1100.400+0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "3f1f500312161f1ae067abe07d13b40f78f32e07"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.8"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "44c37b4636bc54afac5c574d2d02b625349d6582"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.41.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "ae02104e835f219b8930c7664b8012c93475c340"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.2"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "dd933c4ef7b4c270aacd4eb88fa64c147492acf0"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.10.0"

[[Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "80ced645013a5dbdc52cf70329399c35ce007fae"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.13.0"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "1bd6fc0c344fc0cbee1f42f8d2e7ec8253dda2d2"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.25"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "1c5a84319923bea76fa145d49e93aa4394c73fc2"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.1"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "9a5c62f231e5bba35695a20988fc7cd6de7eeb5a"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.3"

[[ImageIO]]
deps = ["FileIO", "JpegTurbo", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "464bdef044df52e6436f8c018bea2d48c40bb27b"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.1"

[[ImageShow]]
deps = ["Base64", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "d0ac64c9bee0aed6fdbb2bc0e5dfa9a3a78e3acc"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.3"

[[Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "a77b273f1ddec645d1b7c4fd5fb98c8f90ad10a5"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.1"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[JuMP]]
deps = ["Calculus", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MathOptInterface", "MutableArithmetics", "NaNMath", "OrderedCollections", "Printf", "Random", "SparseArrays", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "fe0f87cc077fc6a23c21e469318993caf2947d10"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "0.22.3"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "e5718a00af0ab9756305a0392832c8952c7426c1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.6"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[METIS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "1d31872bb9c5e7ec1f618e8c4a56c8b0d9bddc7e"
uuid = "d00139f3-1899-568f-a2f0-47f597d42d70"
version = "5.1.1+0"

[[MUMPS_seq_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "METIS_jll", "OpenBLAS32_jll", "Pkg"]
git-tree-sha1 = "29de2841fa5aefe615dea179fcde48bb87b58f57"
uuid = "d7ed1dd3-d0ae-5e8e-bfb4-87a502085b8d"
version = "5.4.1+0"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "JSON", "LinearAlgebra", "MutableArithmetics", "OrderedCollections", "Printf", "SparseArrays", "Test", "Unicode"]
git-tree-sha1 = "e8c9653877adcf8f3e7382985e535bb37b083598"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "0.10.9"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "842b5ccd156e432f369b204bb704fd4020e383ac"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "0.3.3"

[[NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[Netpbm]]
deps = ["FileIO", "ImageCore"]
git-tree-sha1 = "18efc06f6ec36a8b801b23f076e3c6ac7c3bf153"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.0.2"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "043017e0bdeff61cfbb7afeb558ab29536bbb5ed"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.8"

[[OpenBLAS32_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ba4a8f683303c9082e84afba96f25af3c7fb2436"
uuid = "656ef2d0-ae68-5445-9ca0-591084a874a2"
version = "0.3.12+1"

[[OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "923319661e9a22712f24596ce81c54fc0366f304"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.1+0"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[Osi_jll]]
deps = ["Artifacts", "CoinUtils_jll", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS32_jll", "Pkg"]
git-tree-sha1 = "28e0ddebd069f605ab1988ab396f239a3ac9b561"
uuid = "7da25872-d9ce-5375-a4d3-7a845f58efdd"
version = "0.10800.600+0"

[[PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "eb4dbb8139f6125471aa3da98fb70f02dc58e49c"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.14"

[[PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "03a7a85b76381a3d04c7a1656039197e70eda03d"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.11"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "13468f237353112a01b2d6b32f3d0f80219944aa"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.2"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "a7a7e1a88853564e551e4eba8650f8c38df79b37"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.1.1"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8979e9802b4ac3d58c503a20f2824ad67f9074dd"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.34"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "db3a23166af8aebf4db5ef87ac5b00d36eb771e2"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.0"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "2cf929d64681236a2e074ffafb8d568733d2e6af"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.3"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "8fb59825be681d451c246a795117f317ecbcaa28"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.2"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "8d0c8e3d0ff211d9ff4a0c2307d876c99d10bdf1"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.2"

[[StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "95c6a5d0e8c69555842fc4a927fc485040ccc31c"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.3.5"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c3d8ba7f3fa0625b062b82853a7d5229cb728b6b"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.1"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "bb1064c9a84c52e277f1096cf41434b675cd368b"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.1"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "991d34bbff0d9125d93ba15887d6594e8e84b305"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.5.3"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "78736dab31ae7a53540a6b752efc61f77b304c5b"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.8.6+1"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─d44f3f70-705b-43a2-85c0-a441e08fbbe6
# ╟─a12d1135-ce89-414c-866e-14b4415e1990
# ╟─d35c109a-e5b6-4c01-a9c2-333b883783f2
# ╠═2d6d0588-f4e4-4cad-8353-9c0ba020fec6
# ╟─2877f488-1ab3-46da-8a49-7db65c3d8a45
# ╠═cf6d3f77-781d-4cd0-ad45-9c5c58380354
# ╟─11c8323a-9cd5-4580-93d9-ebcd085b4551
# ╠═0a9212cd-4c79-4752-930b-45dce6a1f3e8
# ╟─7f7439d3-3d3b-4098-8be5-ca9aa59373ce
# ╠═3e708ffa-05d9-4cf7-ba4c-11ca5a3e43e7
# ╟─54082ac4-c31d-4d27-9cc3-5248018c6838
# ╠═6bc42cc7-e3d4-4470-bece-6af950280372
# ╟─a5f3abe7-a843-4ea5-8e77-dfd075b243a5
# ╠═ee41a6b3-fb13-4269-ab22-ad2a85bf26e5
# ╟─dd262e8b-4697-4aa7-9bc7-a32ac25c96dc
# ╠═d5fc7ca7-33cd-4f8a-aecf-8897bcd40e15
# ╟─4ab84877-f5d8-4c1b-b0d2-346f0048bb20
# ╠═1054b6a4-1c28-46b6-88a4-6f7cae84edee
# ╟─07f58656-4826-4ce5-8c97-bf5570f0fc60
# ╠═17088a0b-a568-431c-8d01-13f957e2a4c0
# ╟─4ded521e-c41f-4927-854b-86f8912269db
# ╠═dab4e559-9dc6-4a72-8d38-9a0487b6251a
# ╟─aa340dc7-8833-4e40-b63e-c602096e40e3
# ╟─a2fad4bc-66fa-4466-ae29-b495590f035f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
