import cytnx

def conc(tensors, contractions):
    """
    tensors: list of tuples, each tuple contains the name and labels of a tensor
    contractions: list of tuples, each tuple contains the names of the tensors to be contracted
    """
    # print(">"*80)
    # print(">>>>>Contraction of tensors")
    # print(">"*80)

    # add tensor name to to labels to make them unique
    T_labels = {}
    T_labels_remaining = {}
    for i, (name, labels) in enumerate(tensors):
        T_labels[name] = [name+'_'+label for label in labels]
        T_labels_remaining[name] = labels.copy()

    # print(">>>>>New Labels") 
    # for name in T_labels:
    #     print(name, T_labels[name])
    # print(">>>>>Remaining Labels") 
    # for name in T_labels_remaining:
    #     print(name, T_labels_remaining[name])

    # re-name labels in contraction pair
    # find remaining labels in T_labels_remaining
    # print(">>>>>Contractions:")
    for i, (name_1, label_1, name_2, label_2) in enumerate(contractions):
        # print(">>>>>Pair-{}".format(i))

        # print(name_1, label_1, name_2, label_2)
        label_pair = name_1+'_'+label_1+'-'+name_2+'_'+label_2
        # print(label_pair)

        label_index_1 = T_labels[name_1].index(name_1+'_'+label_1)
        # print(T_labels[name_1][label_index_1])
        T_labels[name_1][label_index_1] = label_pair

        label_index_1 = T_labels_remaining[name_1].index(label_1)
        # print(label_index_1, T_labels_remaining[name_1][label_index_1])
        T_labels_remaining[name_1].pop(label_index_1)

        label_index_2 = T_labels[name_2].index(name_2+'_'+label_2)
        # print(T_labels[name_1][label_index_1])
        T_labels[name_2][label_index_2] = label_pair

        label_index_2 = T_labels_remaining[name_2].index(label_2)
        # print(label_index_2, T_labels_remaining[name_2][label_index_2])
        T_labels_remaining[name_2].pop(label_index_2)

        # print(">>>>>New Labels") 
        # for name in T_labels:
        #     print(name, T_labels[name])
        # print(">>>>>Remaining Labels") 
        # for name in T_labels_remaining:
        #     print(name, T_labels_remaining[name])

    # print(">>>>>Constructing network string")
    # print(">>>>>Net_string:")
    net_string = []
    for name, _ in tensors:
        # print(name+': '+', '.join(T_labels[name]))
        net_string.append(name+': '+', '.join(T_labels[name]))

    # print(">>>>>Tout_labels:")
    Tout_labels = []
    for name in T_labels_remaining.keys():
        Tout_labels += [name+'_'+label for label in T_labels_remaining[name]]
    # print('TOUT: '+', '.join(Tout_labels))
    net_string.append('TOUT: '+', '.join(Tout_labels))
    net=cytnx.Network()
    net.FromString(net_string)

    return net, net_string

