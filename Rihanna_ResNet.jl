using MLDatasets
using Flux
using Flux: Data.DataLoader
using Flux: @epochs, onehotbatch, crossentropy, Momentum, update!, onecold, ADAM
using Plots
using CUDA
using Statistics

train_x, train_y = CIFAR10.traindata(Float32)
test_x, test_y = CIFAR10.testdata(Float32)

# Combinator 
struct Combinator
    conv::Chain
end |> gpu
Combinator(input_filter, filter) = Combinator(shortcut(input_filter, filter))

# Combinator Function 
function (op::Combinator)(x, y)
    z = op.conv(y)
    return x + z
end

# IdentityBlock
identityBlock(input_filter, filter) = Chain(
    # layer 1
    Conv((3, 3), input_filter => filter, pad=(1, 1), stride=(1, 1)),
    BatchNorm(filter, relu),

    # layer 2
    Conv((3, 3), filter => filter, pad=(1, 1), stride=(1, 1)),
    BatchNorm(filter, relu),
    
    # layer 3
    Conv((3, 3), filter => filter, pad=(1, 1), stride=(1, 1)),  
    BatchNorm(filter, relu),    
) |> gpu

# Convolutional Block.
convBlock(input_filter, filter) = Chain(
    # layer 1
    Conv((3, 3), input_filter => filter, pad=(1, 1), stride=(1, 1)), 
    BatchNorm(filter, relu),

    # layer 2
    Conv((3, 3), filter => filter, pad=(1, 1), stride=(1, 1)),
    BatchNorm(filter, relu),
    
    # layer 3
    Conv((3, 3), filter => filter, stride=(1, 1)), # pad=0 is valid
    BatchNorm(filter),
) |> gpu

# Shortcut. 
shortcut(input_filter, filter) = Chain(
  Conv((3,3), input_filter=>filter, stride = (1,1)), # pading = 0 is valid
  BatchNorm(filter),
) |> gpu


model = Chain(
    # Input size = 32*32*3*1 
    
    # Prepare for ResNet
    Conv((7,7), 3=>64, stride=(2,2)), # output = 13*13*64 : params = 3*7*7*64: 
    BatchNorm(64, relu),
    MaxPool((3,3), pad=(1,1), stride=(2,2)), # output = 7*7*64
    
    # ResNet 18 layer: The simplest "ResNet"-type connection is just SkipConnection(layer, +)
    SkipConnection(convBlock(64, 64), Combinator(64, 64)), # output = 5*5*64

    # IdentityBlock must have the same input_filter and filter size. Otherwiese it gives error identityBlock(64, 128) -> error
    SkipConnection(identityBlock(64, 64), +), # output = 5*5*64
    SkipConnection(identityBlock(64, 64), +), # output = 5*5*64
    SkipConnection(convBlock(64, 128), Combinator(64, 128)), # output = 3*3*128
    SkipConnection(identityBlock(128, 128), +), # output = 3*3*128
    SkipConnection(identityBlock(128, 128), +), # output = 3*3*128

    # full connections layers
    MaxPool((2, 2)), # 1*1*128
    Flux.flatten, 
    Dropout(0.3),    
    Dense(128, 1024, relu), 
    Dropout(0.3),
    Dense(1024, 10),
    softmax
) |> gpu

# Data Preparation.
# CUDA tensor - GPU Version.
train_x_tensor = permutedims(train_x, [1, 2, 3, 4])
train_y_onehot = onehotbatch(train_y, 0:9)

test_x_tensor = permutedims(test_x, [1, 2, 3, 4])
test_y_onehot = onehotbatch(test_y, 0:9)

cu_train_x_tensor = cu(train_x_tensor)
cu_train_y_onehot = cu(train_y_onehot)

cu_test_x_tensor = cu(test_x_tensor)
cu_test_y_onehot = cu(test_y_onehot)

train_data = DataLoader((cu_train_x_tensor, cu_train_y_onehot), batchsize=50, shuffle=true) # 100 blocks
test_data = DataLoader((cu_test_x_tensor, cu_test_y_onehot), batchsize=10) # 100 blocks

@info("Conversion is done")

# Optimiser. 
lr = 0.001 # learning_rate
opt = ADAM(lr, (0.9, 0.999)) 

# Loss/Accuracy Functions. 
ϵ = 1.0f-32
loss(x, y) = sum(crossentropy(model(x), y))
accuracy(x, y) = mean(onecold(model(x), 1:10) .== onecold(y, 1:10))

# Callbacks. 
train_losses = []
test_losses = []
train_acces = []
test_acces = []


function loss_all(data_loader)
    sum([loss(x, y) for (x,y) in data_loader]) / length(data_loader) 
end

function acc(data_loader) 
    sum([accuracy(x, y) for (x,y) in data_loader]) / length(data_loader)
end


# Training Function. 
# epochs = 50 # GPU must be on. CPU takes a lot of time for each epoch
epochs = 1


best_acc = 0.0 
for epoch = 1:epochs
    @info "epoch" epoch
	@time begin
	num = 0 
	l = 0
	Flux.train!(loss, params(model), train_data, opt)

	@info "...Calculating..."
	push!(train_losses, loss_all(train_data))
    push!(test_losses, loss_all(test_data)),

    push!(train_acces, acc(train_data)),
    push!(test_acces, acc(test_data))  

	@info("Show")
	@show train_loss = loss_all(train_data)
	@show test_loss = loss_all(test_data)
	@show train_acc = acc(train_data)
	@show test_acc = acc(test_data)

	end
end

# Plot results. 
plot([train_losses, test_losses], title = "Loss", label = ["Training" "Test"])
plot([train_acces, test_acces], title = "Accuracy", label = ["Training" "Test"])
