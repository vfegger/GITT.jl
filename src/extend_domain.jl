import DomainSets
import StaticArrays


# Expand Domain functionality with a function to create create a Vector of n-1 dimensional faces for HyperRectangles
function normed_boundary(D::DomainSets.UnitInterval{T}) where {T}
    left = DomainSets.leftendpoint(D)
    right = DomainSets.rightendpoint(D)

    faces = [
        DomainSets.Point(left),
        DomainSets.Point(right)
    ]

    normals = [
        StaticArrays.SVector(-1.0),
        StaticArrays.SVector(1.0)
    ]

    return faces, normals
end
function normed_boundary(D::DomainSets.HyperRectangle{<:StaticArrays.SVector{2,T}}) where {T}
    left = DomainSets.leftendpoint(D)
    right = DomainSets.rightendpoint(D)
    x1 = left[1]
    x2 = right[1]
    y1 = left[2]
    y2 = right[2]

    d_unit = DomainSets.UnitInterval{T}()
    maps = [
        DomainSets.cube_face_map(zero(T), one(T), StaticArrays.SVector(x1, y1), StaticArrays.SVector(x2, y1)),
        DomainSets.cube_face_map(zero(T), one(T), StaticArrays.SVector(x2, y1), StaticArrays.SVector(x2, y2)),
        DomainSets.cube_face_map(zero(T), one(T), StaticArrays.SVector(x2, y2), StaticArrays.SVector(x1, y2)),
        DomainSets.cube_face_map(zero(T), one(T), StaticArrays.SVector(x1, y2), StaticArrays.SVector(x1, y1))
    ]
    faces = map(m -> DomainSets.ParametricDomain(m, d_unit), maps)
    normals = [
        StaticArrays.SVector(0.0, -1.0),
        StaticArrays.SVector(1.0, 0.0),
        StaticArrays.SVector(0.0, 1.0),
        StaticArrays.SVector(-1.0, 0.0)
    ]

    return faces, normals
end
function normed_boundary(D::DomainSets.HyperRectangle{<:StaticArrays.SVector{N,T}}) where {N,T}
    left2 = DomainSets.leftendpoint(D)
    right2 = DomainSets.rightendpoint(D)
    d_unit = DomainSets.UnitCube{StaticArrays.SVector{N - 1,T}}()
    left1 = DomainSets.leftendpoint(d_unit)
    right1 = DomainSets.rightendpoint(d_unit)
    map1 = DomainSets.cube_face_map(left1, right1, left2, right2, 1, left2[1])
    MAP = typeof(map1)
    maps = MAP[]
    normals = StaticArrays.SVector{N,T}[]
    for dim in 1:N
        push!(maps, DomainSets.cube_face_map(left1, right1, left2, right2, dim, left2[dim]))
        push!(normals, StaticArrays.SVector{N,T}(ntuple(i -> i == dim ? -1.0 : 0.0, N)))
        push!(maps, DomainSets.cube_face_map(left1, right1, left2, right2, dim, right2[dim]))
        push!(normals, StaticArrays.SVector{N,T}(ntuple(i -> i == dim ? 1.0 : 0.0, N)))
    end
    faces = map(m -> DomainSets.ParametricDomain(m, d_unit), maps)

    return faces, normals
end
function normed_boundary(D::DomainSets.HyperRectangle{Vector{T}}) where {T}
    if dimension(D) == 2
        left = DomainSets.leftendpoint(D)
        right = DomainSets.rightendpoint(D)
        x1 = left[1]
        y1 = left[2]
        x2 = right[1]
        y2 = right[2]
        d_unit = DomainSets.UnitInterval{T}()
        maps = [
            DomainSets.cube_face_map(zero(T), one(T), [x1, y1], [x2, y1]),
            DomainSets.cube_face_map(zero(T), one(T), [x2, y1], [x2, y2]),
            DomainSets.cube_face_map(zero(T), one(T), [x2, y2], [x1, y2]),
            DomainSets.cube_face_map(zero(T), one(T), [x1, y2], [x1, y1])
        ]
        normals = [
            [0.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0]
        ]
    else
        left2 = DomainSets.leftendpoint(D)
        right2 = DomainSets.rightendpoint(D)
        d_unit = DomainSets.UnitCube(dimension(D) - 1)
        left1 = DomainSets.leftendpoint(d_unit)
        right1 = DomainSets.rightendpoint(d_unit)

        map1 = DomainSets.cube_face_map(left1, right1, left2, right2, 1, left2[1])
        MAP = typeof(map1)
        maps = MAP[]
        normals = Vector{T}[]
        for dim in 1:dimension(D)
            push!(maps, DomainSets.cube_face_map(left1, right1, left2, right2, dim, left2[dim]))
            push!(normals, [ntuple(i -> i == dim ? -1.0 : 0.0, dimension(D))...])
            push!(maps, DomainSets.cube_face_map(left1, right1, left2, right2, dim, right2[dim]))
            push!(normals, [ntuple(i -> i == dim ? 1.0 : 0.0, dimension(D))...])
        end
    end
    faces = map(m -> DomainSets.ParametricDomain(m, d_unit), maps)

    return faces, normals
end