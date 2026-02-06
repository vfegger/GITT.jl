using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Test
include("../src/extend_domain.jl")


@testset "Extend DomainSets.jl functionality" begin
    D1 = DomainSets.UnitInterval()
    faces, normals = normed_boundary(D1)
    @test length(faces) == 2
    @test length(normals) == 2

    expected_normals = [
        [-1.0],
        [1.0]
    ]
    for n in expected_normals
        @test n in normals
    end
    
    D2 = DomainSets.UnitSquare()
    faces, normals = normed_boundary(D2)

    @test length(faces) == 4
    @test length(normals) == 4

    expected_normals = [
        [0.0, -1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0]
    ]

    for n in expected_normals
        @test n in normals
    end

    D3 = DomainSets.UnitCube()
    faces3, normals3 = normed_boundary(D3)
    @test length(faces3) == 6
    @test length(normals3) == 6

    expected_normals3 = [
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0]
    ]
    for n in expected_normals3
        @test n in normals3
    end
end