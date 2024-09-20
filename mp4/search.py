import heapq

def best_first_search(starting_state):
    # TODO(III): You should copy your code from MP3 here
    visited_states = {starting_state:(None, 0)}

    frontier = []
    heapq.heappush(frontier, starting_state)

    while frontier != []:
            current_state = heapq.heappop(frontier)
            if (current_state.is_goal()):
                return backtrack(visited_states, current_state)
            for neighbor in current_state.get_neighbors():
                current_state.dist_from_start + neighbor.h
                if neighbor not in visited_states:
                    heapq.heappush(frontier, neighbor)
                    visited_states[neighbor] = (current_state, neighbor.dist_from_start)
                elif visited_states[neighbor][1] > neighbor.dist_from_start:
                    visited_states[neighbor] = (current_state, neighbor.dist_from_start)
    return []

def backtrack(visited_states, goal_state):
    # TODO(III): You should copy your code from MP3 here
    path = []

    current_state = goal_state
    while current_state:
        path.append(current_state)
        current_state = visited_states[current_state][0]

    print(path)
    return path[::-1]