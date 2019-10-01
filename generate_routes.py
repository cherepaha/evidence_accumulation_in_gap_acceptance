import numpy as np

def get_path(n_intersections=25):
    # p_left, p_straight, p_right
    turn_probabilities = np.array([0.6, 0.2, 0.2])

    n_turns = np.floor(n_intersections*turn_probabilities).astype(int)

    # 1 is left turn, 0 is go straight, -1 is right turn
    path = np.concatenate([np.ones(n_turns[0]), np.zeros(n_turns[1]), -np.ones(n_turns[2])])

    np.random.shuffle(path)

    print(path)
    return path

def rotate(vector, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.squeeze(np.asarray(np.dot(np.matrix([[c, -s], [s, c]]), vector)))

start_point = np.array([0, 0])
start_direction = np.array([1, 0])

n_paths = 2

for i in range(n_paths):
    valid_path = False
    j = 0
    while not (valid_path):
        print(j)
        j+=1
        locations = [start_point]
        directions = [start_direction]

        path = get_path()

        for k, turn in enumerate(path):
            new_location = locations[-1] + directions[-1]
            new_direction = rotate(directions[-1], turn*np.pi/2)
            locations.append(new_location.astype(int))
            directions.append(new_direction.astype(int))

        locations = np.array(locations)

        valid_path = (locations>-1).all() & (locations<5).all()

    np.savetxt('route_%i.txt' % (i+1), path, fmt='%i')