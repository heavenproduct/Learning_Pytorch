# 这里是对于关键字参数的一个小实验
def vgg(features, num_classes=1000, init_weights=True):
    print('features', features)
    print('num_classes', num_classes)
    print('init_weights', init_weights)

features = 'Hello'

kwargs_1 = {'num_classes': 5,
          'init_weights': True}

kwargs_2 = {'num_classes': 5,
          'init_weights': True,
          'time': '10:00'}

kwargs_3 = {'num_classes': 5}

kwargs_4 = {'init_weights': True}

# vgg(features, **kwargs_1)
#
# # vgg(features, **kwargs_2)    # 这里会报错, 字典中的关键字必须和函数定义中的完全一致(多了不行)
#
# vgg(features, **kwargs_3)    # 这里不会报错, 因为有默认值

# vgg(**kwargs_1)    # 这里因为没有定义 features 报错
# vgg(**kwargs_2)    # 这里因为多出了一个'time', 没有和原函数一一对应报错

# def vgg2(features, num_classes, init_weights=True):
#     print('features', features)
#     print('num_classes', num_classes)
#     print('init_weights', init_weights)
#
# vgg2(features, **kwargs_4)    # 这里因为没有定义 num_classes, 并且在函数中没有定义默认值,报错