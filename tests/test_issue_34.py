from hyperactive import Hyperactive

search_space = dict(my_para=range(100, 200, 10), my_para_2=[0.1, 0.2, 0.3])


def obj_func(para):
    score = 1
    my_para = para["my_para"]
    print("my_para", my_para)
    for i in range(my_para):
        pass
    return score


hyper = Hyperactive()
hyper.add_search(obj_func, search_space, n_iter=100)
hyper.run()
