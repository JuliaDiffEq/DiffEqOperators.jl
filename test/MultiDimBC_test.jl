using LinearAlgebra, DiffEqOperators, Random, Test
################################################################################
# Test 2d extension
################################################################################

#Create Array
n = 100
m = 120
A = rand(n,m)

#Create atomic BC
q1 = RobinBC([1.0, 2.0, 3.0], [0.0, -1.0, 2.0], [0.1, 0.1], 4.0)
q2 = PeriodicBC{Float64}()

BCx = vcat(fill(q1, div(m,2)), fill(q2, div(m,2)))  #The size of BCx has to be all size components *except* for x
BCy = vcat(fill(q1, div(n,2)), fill(q2, div(n,2)))


Qx = MultiDimBC(BCx, 1)
Qy = MultiDimBC(BCy, 2)

Ax = Qx*A
Ay = Qy*A

@test size(Ax)[1] == size(A)[1]+2
@test size(Ay)[2] == size(A)[2]+2

for j in 1:m
    @test Ax[:, j]  == Array(BCx[j]*A[:, j])
end
for i in 1:n
    @test Ay[i,:] == Array(BCy[i]*A[i,:])
end


################################################################################
# Test 3d extension
################################################################################

#Create Array
n = 100
m = 120
o = 78
A = rand(n,m, o)

#Create atomic BC
q1 = RobinBC([1.0, 2.0, 3.0], [0.0, -1.0, 2.0], [0.1, 0.1], 4.0)
q2 = PeriodicBC{Float64}()

BCx = vcat(fill(q1, (div(m,2), o)), fill(q2, (div(m,2), o)))  #The size of BCx has to be all size components *except* for x
BCy = vcat(fill(q1, (div(n,2), o)), fill(q2, (div(n,2), o)))
BCz = fill(MixedBC(q1,q2), (n,m))

Qx = MultiDimBC(BCx, 1)
Qy = MultiDimBC(BCy, 2)
Qz = MultiDimBC(MixedBC(q1, q2), size(A), 3) #Test the other constructor

Ax = Qx*A
Ay = Qy*A
Az = Qz*A
# Test padded array compositions
Aextended = compose(Ax,Ay,Az)
Aflipextended = compose(Az,Ax,Ay)
@test_broken compose(Ax, Az, Az)

#test BC compositions
Q = compose(Qx,Qy,Qz)
@test_broken compose(Qx, Qx, Qz)
Qflip = compose(Qz, Qy, Qx)
QA = Q*A
QflipA = Qflip*A
for i in 1:(n+2), j in 1:(m+2), k in 1:(o+2)
    @test QA[i,j,k] = QFlipA[i,j,k]
    @test Aextended[i,j,k] == QA[i,j,k]
    @test Aextended[i,j,k] == Aflipextended[i,j,k]
end

@test size(Ax)[1] == size(A)[1]+2
@test size(Ay)[2] == size(A)[2]+2
@test size(Az)[3] == size(A)[3]+2
for j in 1:m, k in 1:o
    @test Ax[:, j, k] == Array(BCx[j, k]*A[:, j, k])
end
for i in 1:n, k in 1:o
    @test Ay[i, :, k] == Array(BCy[i, k]*A[i, :, k])
end
for i in 1:n, j in 1:m
    @test Az[i, j, :] == Array(BCz[i, j]*A[i, j, :])
end
