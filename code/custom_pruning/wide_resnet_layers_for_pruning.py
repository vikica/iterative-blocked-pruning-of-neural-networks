# We just hardcode the layers that we want to prune
def get_layers(net):

    selected_layers = [
        net.layer1[0].conv1,
        net.layer1[0].conv2,
        net.layer1[0].shortcut[0],
        net.layer1[1].conv1,
        net.layer1[1].conv2,
        net.layer1[2].conv1,
        net.layer1[2].conv2,
        net.layer1[3].conv1,
        net.layer1[3].conv2,
        net.layer2[0].conv1,
        net.layer2[0].conv2,
        net.layer2[0].shortcut[0],
        net.layer2[1].conv1,
        net.layer2[1].conv2,
        net.layer2[2].conv1,
        net.layer2[2].conv2,
        net.layer2[3].conv1,
        net.layer2[3].conv2,
        net.layer3[0].conv1,
        net.layer3[0].conv2,
        net.layer3[0].shortcut[0],
        net.layer3[1].conv1,
        net.layer3[1].conv2,
        net.layer3[2].conv1,
        net.layer3[2].conv2,
        net.layer3[3].conv1,
        net.layer3[3].conv2
    ]

    return selected_layers
