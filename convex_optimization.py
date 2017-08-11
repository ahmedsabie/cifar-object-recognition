from copy import deepcopy


class GradientDescent:
    ALPHA_MIN = 0.0000001
    ALPHA_MAX = 1.0
    ITERATIONS = 100

    def __init__(self, model):
        self.model = deepcopy(model)

    def find_min(self):
        print 'Iteration: Cost'
        model = deepcopy(self.model)
        theta = model.get_theta()
        for i in xrange(self.ITERATIONS):
            alpha = self.ALPHA_MAX
            updated = False
            while alpha >= self.ALPHA_MIN and not updated:
                previous_cost = model.cost_function()
                try_theta = theta - alpha*model.cost_function_gradient()
                model.set_theta(try_theta)
                new_cost = model.cost_function()
                if new_cost < previous_cost:
                    theta = try_theta
                    updated = True
                else:
                    model.set_theta(theta)
                    alpha /= 2
            print 'Iteration %d: %.2f' % (i+1, model.cost_function())
        return theta
