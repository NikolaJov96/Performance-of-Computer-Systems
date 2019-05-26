import numpy as np
import numpy.random as random
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from math import pow
from math import log


def exp(mean):
    return -log(1 - random.random()) * mean


def init_resources(user_disks_num, jobs_num):
    resources = [
        {
            'avg_proc_time': 20,
            'queue': 0,
            'finish_time': -1,
            'U': 0.0,
            'J': 0.0,
        } for _ in range(1 + 2 + user_disks_num)
    ]

    # Update data for processor and system disks
    resources[0]['avg_proc_time'] = 5
    resources[1]['avg_proc_time'] = 12
    resources[2]['avg_proc_time'] = 15

    # Load jobs into process or queue and start executing first job
    resources[0]['queue'] = jobs_num - 1
    resources[0]['finish_time'] = 0 + exp(resources[0]['avg_proc_time'])
    return resources


def get_soonest_event_resource(resources):
    soonest_res_id = -1
    for i, resource in enumerate(resources):
        if resource['finish_time'] >= 0:
            if soonest_res_id == -1 or resource['finish_time'] < resources[soonest_res_id]['finish_time']:
                soonest_res_id = i
    return soonest_res_id


def get_next_res_id(num_of_resources, source_id):
    if source_id == 0:
        # Source is processor
        rnd = random.random()
        if rnd < 0.15:
            return 1
        elif rnd < 0.3:
            return 2
        else:
            return random.randint(3, num_of_resources)
    elif source_id in [1, 2]:
        # Source is system disk
        if random.random() < 0.5:
            return 0
        else:
            return random.randint(3, num_of_resources)
    else:
        return 0


def sim(user_disks_num, jobs_num, simulation_duration_in_minutes):
    simulation_duration = 1000 * 60 * simulation_duration_in_minutes
    time_elapsed = 0
    num_of_job_laps = 0

    # Define list of resources
    resources = init_resources(user_disks_num, jobs_num)

    while time_elapsed < simulation_duration:
        # Get resource that will finish its job soonest
        soonest_res_id = get_soonest_event_resource(resources)
        soonest_res = resources[soonest_res_id]

        # Calculate elapsed time
        dt = soonest_res['finish_time'] - time_elapsed
        time_elapsed = soonest_res['finish_time']

        # Step stats of each resource
        for resource in resources:
            if resource['finish_time'] >= 0:
                resource['U'] += dt
                resource['J'] += (resource['queue'] + 1) * dt

        # Forward finished job
        next_res_id = get_next_res_id(len(resources), soonest_res_id)
        resources[next_res_id]['queue'] += 1
        # If job was forwarded to the processor, increment it's count
        num_of_job_laps += 1 if next_res_id == 0 else 0
        soonest_res['finish_time'] = -1

        # Load new job from queue if possible
        for res_id in [soonest_res_id, next_res_id]:
            if resources[res_id]['finish_time'] == -1 and resources[res_id]['queue'] > 0:
                resources[res_id]['queue'] -= 1
                resources[res_id]['finish_time'] = time_elapsed + exp(resources[res_id]['avg_proc_time'])

    # Finalize resource and job stats
    for resource in resources:
        resource['U'] = resource['U'] / time_elapsed
        resource['J'] = resource['J'] / time_elapsed
    reaction_time = time_elapsed * jobs_num / num_of_job_laps

    return resources, reaction_time


def dump_data1(data, user_disks_num_pool, jobs_num_pool):
    with open('file1.txt', 'w') as out_file:
        for user_disks_num in user_disks_num_pool:
            for jobs_num in jobs_num_pool:
                out_file.write('K = %d, N = %d\n\n' % (user_disks_num, jobs_num))
                for i, resource in enumerate(data[user_disks_num][jobs_num][0]):
                    if i == 0:
                        name = 'processor:\n'
                    elif i < 3:
                        name = 'system disk %d:\n' % (i - 1)
                    else:
                        name = 'user disk %d:\n' % (i - 3)
                    out_file.write(name)
                    out_file.write('utilization: U = %f%%\n' % (resource['U'] * 100))
                    out_file.write('flow: X = %f\n' % (resource['U'] / resource['avg_proc_time']))
                    out_file.write('average jobs: J = %f\n' % resource['J'])
                    out_file.write('\n')
                out_file.write('System reaction time: R = %f\n\n\n' % data[user_disks_num][jobs_num][1])


def g_n(user_disks_num, jobs_num):
    # Setup Gordon-Newell equations
    mat = np.matrix([np.array([0.0 for _ in range(user_disks_num + 3)]) for _ in range(user_disks_num + 3)])
    mat[0, [1, 2]] = 0.5
    mat[0, 3:] = 1.0
    mat[[1, 2], 0] = 0.15
    mat[3:, 0] = 0.7 / user_disks_num
    mat[3:, [1, 2]] = 0.5 / user_disks_num

    # Subtract one to the diagonal
    mat -= np.identity(user_disks_num + 3)

    # Multiply each column with corresponding mi
    mat[:, 0] /= 5
    mat[:, 1] /= 12
    mat[:, 2] /= 15
    mat[:, 3:] /= 20

    # Apply x1 = 1 and solve
    b = -mat[1:, 0]
    reduced_mat = mat[1:, 1:]
    x = np.concatenate((np.ones((1, )), np.asarray(np.linalg.solve(reduced_mat, b)).flatten()))

    # Generate Buzen array
    buzen = np.zeros(jobs_num + 1)
    buzen[0] = 1.0
    for j in range(user_disks_num + 3):
        for i in range(1, jobs_num + 1):
            buzen[i] += x[j] * buzen[i - 1]

    # Calc all the stats
    u = x * buzen[-2] / buzen[-1]
    x_flow = np.array([u[0] / 5, u[1] / 12, u[2] / 15] + [u[i + 3] / 20 for i in range(user_disks_num)])
    j = [
        sum([
            pow(x[i], j) * buzen[jobs_num - j] / buzen[jobs_num] for j in range(1, jobs_num)
        ]) for i in range(user_disks_num + 3)
    ]
    r = jobs_num / (0.5 * (x_flow[1] + x_flow[2]) + user_disks_num * x_flow[3])

    return x, u, x_flow, j, r


def dump_data2(data, user_disks_num_pool, jobs_num_pool):
    with open('file2.txt', 'w') as out_file:
        for user_disks_num in user_disks_num_pool:
            for jobs_num in jobs_num_pool:
                out_file.write('K = %d, N = %d\n' % (user_disks_num, jobs_num))
                out_file.write('normalized demand: ')
                out_file.write(', '.join(data[user_disks_num][jobs_num][0].astype('string')))
                out_file.write('\n')
            out_file.write('\n')


def dump_data3(data, user_disks_num_pool, jobs_num_pool):
    with open('file3.txt', 'w') as out_file:
        for user_disks_num in user_disks_num_pool:
            for jobs_num in jobs_num_pool:
                out_file.write('K = %d, N = %d\n\n' % (user_disks_num, jobs_num))
                for i in range(user_disks_num + 3):
                    if i == 0:
                        name = 'processor:\n'
                    elif i < 3:
                        name = 'system disk %d:\n' % (i - 1)
                    else:
                        name = 'user disk %d:\n' % (i - 3)
                    out_file.write(name)
                    out_file.write('utilization: U = %f%%\n' % (data[user_disks_num][jobs_num][1][i] * 100))
                    out_file.write('flow: X = %f\n' % data[user_disks_num][jobs_num][2][i])
                    out_file.write('average jobs: J = %f\n' % data[user_disks_num][jobs_num][3][i])
                    out_file.write('\n')
                out_file.write('System reaction time: R = %f\n\n\n' % data[user_disks_num][jobs_num][4])


def dump_data4(sim_data, analytic_data, user_disks_num_pool, jobs_num_pool):
    with open('file4.txt', 'w') as out_file:
        with open('file5.txt', 'w') as bare_file:
            for user_disks_num in user_disks_num_pool:
                for jobs_num in jobs_num_pool:
                    out_file.write('K = %d, N = %d\n\n' % (user_disks_num, jobs_num))
                    bare_file.write('K = %d, N = %d\n\n' % (user_disks_num, jobs_num))
                    for i, sim_res in enumerate(sim_data[user_disks_num][jobs_num][0]):
                        if i == 0:
                            name = 'processor:'
                        elif i < 3:
                            name = 'system disk %d:' % (i - 1)
                        else:
                            name = 'user disk %d:' % (i - 3)
                        out_file.write(name + '\n')
                        bare_file.write('\n')
                        out_file.write('utilization: Ua/Us = %f%%\n' %
                                       ((analytic_data[user_disks_num][jobs_num][1][i] / sim_res['U'] - 1) * 100))
                        bare_file.write('%f\n' %
                                        ((analytic_data[user_disks_num][jobs_num][1][i] / sim_res['U'] - 1) * 100))
                        out_file.write('flow: Xa/Xs = %f%%\n' %
                                       ((analytic_data[user_disks_num][jobs_num][2][i] /
                                         (sim_res['U'] / sim_res['avg_proc_time']) - 1) * 100))
                        bare_file.write('%f\n' %
                                        ((analytic_data[user_disks_num][jobs_num][2][i] /
                                          (sim_res['U'] / sim_res['avg_proc_time']) - 1) * 100))
                        out_file.write('average jobs: Ja/Js = %f%%\n' %
                                       ((analytic_data[user_disks_num][jobs_num][3][i] / sim_res['J'] - 1) * 100))
                        bare_file.write('%f\n' %
                                        ((analytic_data[user_disks_num][jobs_num][3][i] / sim_res['J'] - 1) * 100))
                        out_file.write('\n')
                        bare_file.write('\n')
                    out_file.write('System reaction time: Ra/Rs = %f%%\n\n\n' %
                                   ((analytic_data[user_disks_num][jobs_num][4] /
                                    sim_data[user_disks_num][jobs_num][1] - 1) * 100))
                    bare_file.write('%f\n\n\n' %
                                    ((analytic_data[user_disks_num][jobs_num][4] /
                                      sim_data[user_disks_num][jobs_num][1] - 1) * 100))


def plot(resource, user_disks_num_pool, jobs_num_pool, data, title_label, metric):
    resource_names = ['processor', 'system disk 0', 'system disk 1', 'user disks']
    metrics = ['Utilization percentage', 'Flow per ms', 'Average number of jobs', 'Reaction time']
    flags = ['U', 'X', 'J', 'R']
    if resource >= 0:
        title = '%s - %s - %s' % (title_label, resource_names[resource if resource < 3 else 3], flags[metric])
    else:
        title = '%s - %s' % (title_label, flags[metric])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title + '\n')
    ax.set_xlabel('Number of resources')
    ax.set_ylabel('Number of jobs')
    ax.set_zlabel(metrics[metric])
    x_axis = [3 + i for _ in range(len(jobs_num_pool)) for i in user_disks_num_pool]
    y_axis = [i for i in jobs_num_pool for _ in range(len(user_disks_num_pool))]
    ax.bar(x_axis, data, y_axis, zdir='y', alpha=0.8)
    fig.savefig(title + '.png')
    fig.clear()


def draw_graphs(sim_data, analytic_data, user_disks_num_pool, jobs_num_pool):
    for resource in range(4):
        # Simulation data
        u_sim = np.array([
            [
                sim_data[user_disks_num][jobs_num][0][resource]['U'] * 100 for user_disks_num in user_disks_num_pool
            ] for jobs_num in jobs_num_pool
        ]).reshape((1, len(jobs_num_pool) * len(user_disks_num_pool)))[0]
        plot(resource, user_disks_num_pool, jobs_num_pool, u_sim, 'Simulation', 0)
        x_sim = np.array([
            [
                sim_data[user_disks_num][jobs_num][0][resource]['U'] /
                sim_data[user_disks_num][jobs_num][0][resource]['avg_proc_time'] for user_disks_num in user_disks_num_pool
            ] for jobs_num in jobs_num_pool
        ]).reshape((1, len(jobs_num_pool) * len(user_disks_num_pool)))[0]
        plot(resource, user_disks_num_pool, jobs_num_pool, x_sim, 'Simulation', 1)
        j_sim = np.array([
            [
                sim_data[user_disks_num][jobs_num][0][resource]['J'] for user_disks_num in user_disks_num_pool
            ] for jobs_num in jobs_num_pool
        ]).reshape((1, len(jobs_num_pool) * len(user_disks_num_pool)))[0]
        plot(resource, user_disks_num_pool, jobs_num_pool, j_sim, 'Simulation', 2)

        # Analytical data
        u_ana = np.array([
            [
                analytic_data[user_disks_num][jobs_num][1][resource] * 100 for user_disks_num in user_disks_num_pool
            ] for jobs_num in jobs_num_pool
        ]).reshape((1, len(jobs_num_pool) * len(user_disks_num_pool)))[0]
        plot(resource, user_disks_num_pool, jobs_num_pool, u_ana, 'Analytical', 0)
        x_ana = np.array([
            [
                analytic_data[user_disks_num][jobs_num][2][resource] for user_disks_num in user_disks_num_pool
            ] for jobs_num in jobs_num_pool
        ]).reshape((1, len(jobs_num_pool) * len(user_disks_num_pool)))[0]
        plot(resource, user_disks_num_pool, jobs_num_pool, x_ana, 'Analytical', 1)
        j_ana = np.array([
            [
                analytic_data[user_disks_num][jobs_num][3][resource] for user_disks_num in user_disks_num_pool
            ] for jobs_num in jobs_num_pool
        ]).reshape((1, len(jobs_num_pool) * len(user_disks_num_pool)))[0]
        plot(resource, user_disks_num_pool, jobs_num_pool, j_ana, 'Analytical', 2)
        
    # Reaction time
    r_sim = np.array([
        [
            sim_data[user_disks_num][jobs_num][1] for user_disks_num in user_disks_num_pool
        ] for jobs_num in jobs_num_pool
    ]).reshape((1, len(jobs_num_pool) * len(user_disks_num_pool)))[0]
    plot(-1, user_disks_num_pool, jobs_num_pool, r_sim, 'Simulation', 3)
    r_ana = np.array([
        [
            analytic_data[user_disks_num][jobs_num][4] for user_disks_num in user_disks_num_pool
        ] for jobs_num in jobs_num_pool
    ]).reshape((1, len(jobs_num_pool) * len(user_disks_num_pool)))[0]
    plot(-1, user_disks_num_pool, jobs_num_pool, r_ana, 'Analytical', 3)
        

def main():
    user_disks_num_pool = range(2, 9)
    jobs_num_pool = [10, 15, 25]
    to_run = ['sim', 'analytic']

    sim_data = {}
    if 'sim' in to_run:
        simulation_duration_in_minutes = 60 * 18
        print 'a) Simulation'
        for user_disks_num in user_disks_num_pool:
            sim_data[user_disks_num] = {}
            for jobs_num in jobs_num_pool:
                print 'User disks: %d, Number of jobs: %d' % (user_disks_num, jobs_num)
                start_time = time.time()
                sim_data[user_disks_num][jobs_num] = sim(user_disks_num, jobs_num, simulation_duration_in_minutes)
                print 'Execution duration: %f\n' % (time.time() - start_time)
        dump_data1(sim_data, user_disks_num_pool, jobs_num_pool)

    analytic_data = {}
    if 'analytic' in to_run:
        print 'b, c) Gordon-Newell'
        for user_disks_num in user_disks_num_pool:
            analytic_data[user_disks_num] = {}
            for jobs_num in jobs_num_pool:
                print 'User disks: %d, Number of jobs: %d' % (user_disks_num, jobs_num)
                start_time = time.time()
                analytic_data[user_disks_num][jobs_num] = g_n(user_disks_num, jobs_num)
                print 'Execution duration: %f\n' % (time.time() - start_time)
        dump_data2(analytic_data, user_disks_num_pool, jobs_num_pool)
        dump_data3(analytic_data, user_disks_num_pool, jobs_num_pool)

    if 'sim' in to_run and 'analytic' in to_run:
        print 'd) sim - analytic relation'
        dump_data4(sim_data, analytic_data, user_disks_num_pool, jobs_num_pool)
        draw_graphs(sim_data, analytic_data, user_disks_num_pool, jobs_num_pool)


if __name__ == '__main__':
    main()
