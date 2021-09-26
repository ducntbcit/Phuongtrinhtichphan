import numpy as np

class Gauss_quadratures:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.gauss_points = []
        self.map_local_to_global = lambda local_point:\
            (a + b) / 2 + (b - a) / 2 * local_point # map numbers from [-1; 1] into [a,b] domain
    def get_global_points(self):
        return [point.global_point for point in self.gauss_points]
    def get_weights(self):
        return [point.weight for point in self.gauss_points ]
    def add_point(self, weight, local_point):
        self.gauss_points.append(Gauss_point(local_point,weight, self.map_local_to_global))

    def gauss_t3_init(self):
        self.gauss_points = []
        self.add_point(8 / 9, 0)
        self.add_point(5 / 9, -np.sqrt(3 / 5))
        self.add_point(5 / 9, np.sqrt(3 / 5))

    def gauss_t20_init(self):
        self.gauss_points = []
        self.add_point(0.1527533871307258, -0.0765265211334973)
        self.add_point(0.1527533871307258, 0.0765265211334973)
        self.add_point(0.1491729864726037, -0.2277858511416451)
        self.add_point(0.1491729864726037, 0.2277858511416451)
        self.add_point(0.1420961093183820, -0.3737060887154195)
        self.add_point(0.1420961093183820, 0.3737060887154195)
        self.add_point(0.1316886384491766, -0.5108670019508271)
        self.add_point(0.1316886384491766, 0.5108670019508271)
        self.add_point(0.1181945319615184, -0.6360536807265150)
        self.add_point(0.1181945319615184, 0.6360536807265150)
        self.add_point(0.1019301198172404, -0.7463319064601508)
        self.add_point(0.1019301198172404, 0.7463319064601508)
        self.add_point(0.0832767415767048, -0.8391169718222188)
        self.add_point(0.0832767415767048, 0.8391169718222188)
        self.add_point(0.0626720483341091, -0.9122344282513259)
        self.add_point(0.0626720483341091, 0.9122344282513259)
        self.add_point(0.0406014298003869, -0.9639719272779138)
        self.add_point(0.0406014298003869, 0.9639719272779138)
        self.add_point(0.0176140071391521, -0.9931285991850949)
        self.add_point(0.0176140071391521, 0.9931285991850949)

class Gauss_point:
    def __init__(self, local_point, weight, mapping_function):
        self.local_point = local_point
        self.weight = weight
        self.global_point = mapping_function(local_point)

# def gauss_t3_init(a,b):
#     quadratures_t3 = Gauss_quadratures(a, b)
#     quadratures_t3.add_point(8/9, 0)
#     quadratures_t3.add_point(5/9, -np.sqrt(3/5))
#     quadratures_t3.add_point(5/9, np.sqrt(3/5))
#     return quadratures_t3
#
# def gauss_t20_init(a,b):
#     quadratures_t20 = Gauss_quadratures(a, b)
#     quadratures_t20.add_point(0.1527533871307258, -0.0765265211334973)
#     quadratures_t20.add_point(0.1527533871307258, 0.0765265211334973)
#     quadratures_t20.add_point(0.1491729864726037, -0.2277858511416451)
#     quadratures_t20.add_point(0.1491729864726037, 0.2277858511416451)
#     quadratures_t20.add_point(0.1420961093183820, -0.3737060887154195)
#     quadratures_t20.add_point(0.1420961093183820, 0.3737060887154195)
#     quadratures_t20.add_point(0.1316886384491766, -0.5108670019508271)
#     quadratures_t20.add_point(0.1316886384491766, 0.5108670019508271)
#     quadratures_t20.add_point(0.1181945319615184, -0.6360536807265150)
#     quadratures_t20.add_point(0.1181945319615184, 0.6360536807265150)
#     quadratures_t20.add_point(0.1019301198172404, -0.7463319064601508)
#     quadratures_t20.add_point(0.1019301198172404, 0.7463319064601508)
#     quadratures_t20.add_point(0.0832767415767048, -0.8391169718222188)
#     quadratures_t20.add_point(0.0832767415767048, 0.8391169718222188)
#     quadratures_t20.add_point(0.0626720483341091, -0.9122344282513259)
#     quadratures_t20.add_point(0.0626720483341091, 0.9122344282513259)
#     quadratures_t20.add_point(0.0406014298003869, -0.9639719272779138)
#     quadratures_t20.add_point(0.0406014298003869, 0.9639719272779138)
#     quadratures_t20.add_point(0.0176140071391521, -0.9931285991850949)
#     quadratures_t20.add_point(0.0176140071391521, 0.9931285991850949)
#     return quadratures_t20