from menu_cost import MenuCostEnv


if __name__ == '__main__':
    env = MenuCostEnv()
    env.vfi()
    env.plot_policy()
