class Net(torch.nn.Module):
    def __init__(self, n_neurons, p):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(n_neurons, 200)
        self.dropout1 = torch.nn.Dropout(p) 
        self.ac1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(200, 100)
        self.dropout2 = torch.nn.Dropout(p) 
        self.ac2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(100, 10)          
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.ac1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.ac2(x)
        x = self.fc3(x)
        return x

def training(data, device, p):
    X_train, y_train, X_test, y_test = data
    net = Net(X_train.shape[1], p)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1.0e-3)
    mnist_net = net.to(device)

    batch_size = 1000
    test_accuracy_history = []
    test_loss_history = []
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    number_epochs = 30
    for epoch in range(number_epochs):
        for start_index in range(0, len(X_train), batch_size):
            optimizer.zero_grad()
            X_batch = X_train[start_index:start_index+batch_size].to(device)
            y_batch = y_train[start_index:start_index+batch_size].to(device)
            preds = net.forward(X_batch) 
            loss_value = loss(preds, y_batch)
            loss_value.backward()
            optimizer.step()
        test_preds = net.forward(X_test)
        test_loss_history.append(loss(test_preds, y_test))
        accuracy = (test_preds.argmax(dim=1) == y_test).float().mean().cpu()
        test_accuracy_history.append(float(accuracy))
        # save weights by layers before and after training
        if epoch == 0 or epoch == number_epochs - 1:
            for param_tensor in net.state_dict():
                if param_tensor == 'fc1.weight':
                    weights1_2 = net.state_dict()[param_tensor].cpu().numpy()
                elif param_tensor == 'fc2.weight':
                    weights2_3 = net.state_dict()[param_tensor].cpu().numpy()
                elif param_tensor == 'fc3.weight':
                    weights3_4 = net.state_dict()[param_tensor].cpu().numpy()
            weights1_2 = weights1_2.reshape((-1))
            weights2_3 = weights2_3.reshape((-1))
            weights3_4 = weights3_4.reshape((-1))
            if epoch == 0:
                df0_12 = pd.DataFrame({'weights': weights1_2, 'type': ['before_training'] * len(weights1_2)})
                df0_23 = pd.DataFrame({'weights': weights2_3, 'type': ['before_training'] * len(weights2_3)})
                df0_34 = pd.DataFrame({'weights': weights3_4, 'type': ['before_training'] * len(weights3_4)})
            else:
                df1_12 = pd.DataFrame({'weights': weights1_2, 'type': ['after_training']*len(weights1_2)})
                df1_23 = pd.DataFrame({'weights': weights2_3, 'type': ['after_training']*len(weights2_3)})
                df1_34 = pd.DataFrame({'weights': weights3_4, 'type': ['after_training']*len(weights3_4)})
    df12 = pd.concat([df0_12, df1_12]).reset_index(drop = True)
    df23 = pd.concat([df0_23, df1_23]).reset_index(drop = True)
    df34 = pd.concat([df0_34, df1_34]).reset_index(drop = True)
    return (df12, df23, df34, net, test_accuracy_history)

def build_graph(net):
    for param_tensor in net.state_dict():
        if param_tensor == 'fc1.weight':
            weights1_2 = net.state_dict()[param_tensor].cpu().numpy()
        elif param_tensor == 'fc2.weight':
            weights2_3 = net.state_dict()[param_tensor].cpu().numpy()
        elif param_tensor == 'fc3.weight':
            weights3_4 = net.state_dict()[param_tensor].cpu().numpy()
    G = nx.Graph()
    
    layer_2,layer_1 = weights1_2.shape    
    layer_3,layer_2 = weights2_3.shape 
    layer_4,layer_3 = weights3_4.shape 
    for i in range(0, layer_1 + layer_2 + layer_3 + layer_4):
        G.add_node(i)

    for j in range(0, layer_2):
        for i in range(0, layer_1):
            G.add_edge(i, layer_1 + j, weight = weights1_2[j][i])

    node_count = layer_1 + layer_2

    for j in range(0, layer_3):
        for i in range(0, layer_2):
            G.add_edge(layer_1 + i, node_count + j, weight = weights2_3[j][i])

    node_count = node_count + layer_3

    for j in range(0, layer_4):
        for i in range(0, layer_3):
            G.add_edge(layer_1 + layer_2 + i,node_count + j, weight = weights3_4[j][i])
    return G

# method of deletion of graph's connections with the weights less than i-th element in linspace 
# and checking network's ability at each step
def percolation(G, net, X_test, y_test, device):
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    net = net.to(device)
    edge_weight = nx.get_edge_attributes(G, 'weight')
    linsp = np.linspace(0, 0.4, 300)
    average_degree = np.array([])
    len_max_connected = np.array([])
    density = np.array([])
    H = G.copy()
    x_accuracy = 0
    history = []
    for i in range(len(linsp)):
        to_remove = [(a,b) for a, b, attrs in H.edges(data=True) if abs(attrs["weight"]) <= linsp[i]]
        shape1 = net.fc1.weight.shape
        shape2 = net.fc2.weight.shape
        shape3 = net.fc3.weight.shape
        for pair in to_remove:
            if pair[0] < shape1[1] and abs(pair[1] - shape1[1]) < shape1[0]:
                with torch.no_grad():
                    net.fc1.weight[pair[1] - shape1[1]][pair[0]] = 0
            elif abs(pair[0] - shape1[1]) < shape1[0] and abs(pair[1] - shape1[1] - shape2[1]) < shape2[0]:
                with torch.no_grad():
                    net.fc2.weight[pair[1] - shape1[1] - shape2[1]][pair[0] - shape1[1]] = 0
            else:
                with torch.no_grad():
                    net.fc3.weight[pair[1] - shape1[1] - shape2[1]- shape3[1]][pair[0] - shape1[1] - shape2[1]] = 0
        H.remove_edges_from(to_remove)
        degrees = [d for n, d in H.degree()]
        average_degree = np.append(average_degree, np.mean(degrees))
        density = np.append(density, nx.density(H))
        LCG_nodes = max(nx.connected_components(H), key=len)
        LCG = G.subgraph(LCG_nodes)
        len_max_connected = np.append(len_max_connected, len(LCG) / len(G))
        if len_max_connected[i] == 1:
            x_component = linsp[i]
        test_preds = net.forward(X_test)
        accuracy = (test_preds.argmax(dim=1) == y_test).float().mean().cpu()
        history.append(accuracy)
        if i != 0 and x_accuracy == 0:
            if abs((history[i - 1] - history[i]) / history[i - 1] * 100) > 0.5:
                x_accuracy = linsp[i - 1]
                y_accuracy = history[i - 1]
    average_degree = average_degree / average_degree[0]
    density = density / density[0]
    
    return linsp, average_degree, density, len_max_connected, x_component, x_accuracy, y_accuracy, history