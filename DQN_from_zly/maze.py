import random
import sys


def two_dimensional_maze(width, height):
    maze = [[-1] * width for i in range(height)]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def dfs(x, y):
        maze[y][x] = 0
        random.shuffle(directions)
        for dx, dy in directions:
            new_x, new_y = (x + 2 * dx + width) % width, (y + 2 * dy + height) % height
            if maze[new_y][new_x] == -1 and maze[(y + dy) % height][(x + dx) % width] != 1:
                maze[(y + dy) % height][(x + dx) % width] = 0
                dfs(new_x, new_y)

    target_x, target_y = random.randint(0, width - 1), random.randint(0, height - 1)
    maze[target_y][target_x] = 1
    random.shuffle(directions)
    dx_, dy_ = directions[0]
    new_x_, new_y_ = (target_x + dx_ + width) % width, (target_y + dy_ + height) % height
    dfs(new_x_, new_y_)
    return maze, (target_x, target_y)


if __name__ == '__main__':
    '''
    用于生成迷宫的代码
    输入迷宫的宽度和高度，生成的迷宫会保存在maze.txt文件中
    
    迷宫的格式为：
    
    第一行：迷宫的高度和宽度，用逗号分隔
    接下来的行：迷宫的每个格子，用空格分隔，0表示通路，-1表示墙，1表示目标点
    
    注意：迷宫的宽度和高度必须为奇数或者同时为偶数，否则会出bug
    '''

    # 注意：r和h必须同时为奇数或者同时为偶数，不然会出bug

    # r, h = int(input('width:')), int(input('height:'))
    r, h = 10, 10

    maze_, target = two_dimensional_maze(r, h)
    sys.stdout = open("maze.txt", "w")
    print(str(h) + "," + str(r))
    for row in maze_:
        print('\t'.join(map(str, row)))

    # print(target)
