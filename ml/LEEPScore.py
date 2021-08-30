import torch


def get_empirical_conditional_distribution(y, z, predict_result, dummy_distribution):
    joint_distribution_yz_numerator = marginal_distribution_z_numerator = 0.0
    n = len(predict_result)
    # 1. Calculate the empirical joint_distribution P(y,z)
    for i in range(n):
        if predict_result[i][1] == y:
            joint_distribution_yz_numerator += dummy_distribution[i][z]
    # 2. Calculate the empirical marginal_distribution P(z)
    for i in range(n):
        marginal_distribution_z_numerator += dummy_distribution[i][z]
    # 3. By rights, both the empirical joint_distribution P(y,z) and the empirical
    # marginal_distribution P(z) should be divided 'n',but the 'n' in the denominator cancel out
    return joint_distribution_yz_numerator / marginal_distribution_z_numerator


def LEEPScore(origin_net, target_data_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    origin_net.to(device)
    # Step 1. calculate the dummy distribution,

    dummy_distribution = []
    predict_result = []

    with torch.no_grad():
        for test_img, test_label in target_data_loader:
            test_img = test_img.to(device)
            test_label = test_label.to(device)

            outputs = origin_net(test_img)
            # predicts = torch.max(outputs, dim=1)[1]
            _, predicts = torch.max(outputs, dim=1)
            outputs = torch.softmax(outputs, dim=1)
            dummy_distribution.extend(outputs)
            # predict_result.append((predicts.item(), test_label.item()))
            predict_result.extend(zip(predicts, test_label))

        else:
            origin_outputs_num = len(outputs[0])
    origin_features, target_features = range(origin_outputs_num), range(len(target_data_loader.dataset.classes))

    # Step 2. calculate the conditional distribution the target label y given the source label z.
    conditional_distribution = {}
    for y in target_features:
        for z in origin_features:
            conditional_distribution[(y, z)] = get_empirical_conditional_distribution(y, z, predict_result,
                                                                                      dummy_distribution)

    # Step 3. Calculate the log-likelihood loss
    n = len(predict_result)
    log_loss = torch.tensor([0.0])
    for i in range(n):
        temp = torch.tensor([0.0])
        for z in origin_features:
            temp += conditional_distribution[(predict_result[i][1].item(), z)] * dummy_distribution[i][z]
        log_loss += torch.log(temp)
    return round((log_loss / n).item(), 4)
