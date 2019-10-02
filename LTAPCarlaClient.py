from pathlib import Path
import glob
import os
import sys
from exp_info_ui import ExpInfoUI

try:
    sys.path.append(glob.glob(os.path.join(str(Path.home()),
    'CarlaLatest/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64')))[0])
    import carla
    import pygame
except BaseException:
    pass

import math
import time
import random
import numpy as np
import tkinter as tkr
import csv

class LTAPCarlaClient():
    block_size = 150
    lane_width = 3.5

    bot_colors = []

#    bot_blueprint_names = ['vehicle.tesla.model3', 'vehicle.nissan.micra', 'vehicle.volkswagen.t2',
#                           'vehicle.mini.cooperst', 'vehicle.citroen.c3']
    bot_blueprint_names = ['vehicle.tesla.model3']

    bot_blueprint_colors = ['147,130,127', '107,162,146', '53,206,141', '188,216,183', '224,210,195',
                        '255,237,101', '180,173,234']

    def __init__(self):
        try:
            self.exp_info = self.get_exp_info()
            self.n_routes_per_session = 4

            self.set_ff_gain()
            self.initialize_log()
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(2.0)
            pygame.init()
            self.world = self.client.get_world()

            self.tta_conditions = [4, 5, 6]
            self.bot_distance_values = [90, 120, 150]

            # x and y indices of the intersection
            # (0, 0) is the intersection in the bottom left corner of the map
            # (4, 4) is the intersection in the top right corner of the map
            self.origin = np.array([0.0, 0.0])

            # (1,0) is east, (-1,0) is west, (0,1) is north, (0,-1) is south
            self.active_intersection = np.array([1.0, 0.0])

            self.sound_cues = {(1, 1): 'next_turn_left',
                               (1, 0): 'next_go_straight',
                               (1, -1): 'next_turn_right',
                               (2, 1): 'turn_left',
                               (2, 0): 'go_straight',
                               (2, -1): 'turn_right'}

            self.ego_actor = None
            self.bot_actor = None
            self.bot_actor_blueprints = [random.choice(self.world.get_blueprint_library().filter(bp_name))
                                            for bp_name in self.bot_blueprint_names]

            self.empty_control = carla.VehicleControl(hand_brake=False, reverse=False, manual_gear_shift = False)
            self.control = self.empty_control

        except KeyboardInterrupt:
            for actor in self.world.get_actors():
                actor.destroy()

    def set_ff_gain(self, gain=35):
        ffset_cmd = 'ffset /dev/input/event%i -a %i'
        for i in range(5,10):
            os.system(ffset_cmd % (i, gain))

    def generate_tta_values(self):
        tta_values = []
        for tta in self.tta_conditions:
            # 5 is the number of left turns per route per tta
            tta_values = np.append(tta_values, np.ones(5)*tta)
        np.random.shuffle(tta_values)

        return tta_values

    def initialize_log(self):
        log_directory = 'data'
        self.log_file_path = os.path.join(log_directory, str(self.exp_info['subj_id']) + '_' +
                                                         str(self.exp_info['session']) + '_' +
                                                         self.exp_info['start_time'] + '.txt')
        with open(self.log_file_path, 'w') as fp:
            writer = csv.writer(fp, delimiter = '\t')
            writer.writerow(['subj_id', 'session', 'route', 'intersection_no',
                             'intersection_x', 'intersection_y','turn_direction', 't',
                             'ego_distance_to_intersection', 'tta_condition', 'd_condition', 'v_condition',
                             'ego_x', 'ego_y', 'ego_vx', 'ego_vy', 'ego_ax', 'ego_ay', 'ego_yaw',
                             'bot_x', 'bot_y', 'bot_vx', 'bot_vy', 'bot_ax', 'bot_ay', 'bot_yaw',
                             'throttle', 'brake', 'steer'])

    def write_log(self, log):
        with open(self.log_file_path, 'a') as fp:
            writer = csv.writer(fp, delimiter = '\t', )
            writer.writerows(log)

    def get_exp_info(self):
        root = tkr.Tk()
        app = ExpInfoUI(master=root)
        app.mainloop()
        exp_info = app.exp_info
        root.destroy()

        return exp_info

    def rotate(self, vector, angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.squeeze(np.asarray(np.dot(np.matrix([[c, -s], [s, c]]), vector)))

    def update_ego_control(self):
        reverse = False
        K1 = 1.0  # 0.55
        steerCmd = K1 * math.tan(1.1 * self.joystick.get_axis(0))

        K2 = 1.6  # 1.6
        throttleCmd = K2 + (2.05 * math.log10(
            -0.7 * self.joystick.get_axis(2) + 1.4) - 1.2) / 0.92
        if throttleCmd <= 0:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1

        # cap the speed at 20 m/s
        speed = np.sqrt(self.ego_actor.get_velocity().x**2 + self.ego_actor.get_velocity().y**2)
        if speed > 20:
            throttleCmd = 0

        brakeCmd = 1.6 + (2.05 * math.log10(
            -0.7 * self.joystick.get_axis(3) + 1.4) - 1.2) / 0.92
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1

        if self.joystick.get_button(5):
            reverse = True
        elif self.joystick.get_button(4):
            reverse = False

        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                if event.key == pygame.locals.K_ESCAPE:
                    raise KeyboardInterrupt

        self.control.throttle = throttleCmd
        self.control.steer = steerCmd
        self.control.brake = brakeCmd
        self.control.reverse = reverse

        self.ego_actor.apply_control(self.control)

    def spawn_ego_car(self):
        '''
        To shift the starting position from the center of the intersection to the lane where
        the driver can start driving towards the first active intersection, we rotate the
        heading direction 90` clockwise (-np.pi/2), and shift the origin towards that direction by half lane width
        '''
        start_position = self.origin*self.block_size + \
                        self.rotate(self.active_intersection-self.origin, -np.pi/2)*self.lane_width/2
        self.ego_start_position = self.world.get_map().get_waypoint(
            carla.Location(x=start_position[0], y=-start_position[1], z=0))

        ego_bp = random.choice(self.world.get_blueprint_library().filter('vehicle.tesla.model3'))

        self.ego_actor = self.world.spawn_actor(ego_bp, self.ego_start_position.transform)
        self.ego_actor.set_autopilot(False)

    def spawn_bot(self, distance_to_intersection, speed):
        bot_bp = random.choice(self.bot_actor_blueprints)
        bot_bp.set_attribute('color', random.choice(self.bot_blueprint_colors))

        ego_direction = self.active_intersection - self.origin

        spawn_location = self.active_intersection*self.block_size + \
                            distance_to_intersection*(ego_direction) + \
                            (self.lane_width/2) * self.rotate(ego_direction, np.pi/2)
        spawn_waypoint = self.world.get_map().get_waypoint(
                        carla.Location(x=spawn_location[0], y=-spawn_location[1], z=0))

        self.bot_actor = self.world.spawn_actor(bot_bp, spawn_waypoint.transform)

        self.bot_velocity = speed*self.rotate(ego_direction, np.pi).astype(int)
        self.bot_actor.set_velocity(carla.Vector3D(self.bot_velocity[0], -self.bot_velocity[1], 0))

    def update_bot_control(self, max_speed):
        self.bot_actor.set_velocity(carla.Vector3D(self.bot_velocity[0], -self.bot_velocity[1], 0))

    def play_sound_cue(self, number, direction):
        sound_filename = '%s.wav' % (self.sound_cues[(number, direction)])
        file_path = os.path.join('sounds', sound_filename)
        sound = pygame.mixer.Sound(file_path)
        sound.set_volume(0.5)
        sound.play()

    def initialize_noise_sound(self):
        file_path = 'sounds/tesla_noise.wav'
        self.noise_sound = pygame.mixer.Sound(file_path)
        self.noise_sound.set_volume(0.1)
        self.noise_sound.play(loops=-1)

    def get_actor_state(self, actor):
        state = ([actor.get_transform().location.x, -actor.get_transform().location.y,
                 actor.get_velocity().x, -actor.get_velocity().y,
                 actor.get_acceleration().x, -actor.get_acceleration().y,
                 actor.get_transform().rotation.yaw]
                 if not (actor is None) else np.zeros(7).tolist())
        return list(['%.4f' % value for value in state])

    def update_log(self, log, values_to_log):
        log.append((values_to_log + \
                    self.get_actor_state(self.ego_actor) + \
                    self.get_actor_state(self.bot_actor) + \
                    list(['%.4f' % value for value in [self.control.throttle, self.control.brake, self.control.steer]])))

    def run(self):
        try:
            print(self.exp_info)
            first_route = self.exp_info['route']

            for i in range(first_route, self.n_routes_per_session+1):
                tta_values = self.generate_tta_values()
                self.origin = np.array([0.0, 0.0])
                self.active_intersection = np.array([1.0, 0.0])

                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()

                self.spawn_ego_car()

                self.initialize_noise_sound()
                # in the first session, we go through routes 1 to 4, in the second session, routes 5 to 8
                route_number = i + (self.exp_info['session']-1)*self.n_routes_per_session
                # in the path input file, -1 is turn right, 1 is turn left, 0 is go straight
                route = np.loadtxt(os.path.join('routes', 'route_%i.txt' % (route_number)))
                tta = tta_values[-1]
                for j, current_turn in enumerate(route):
                    # if the current turn is left, set TTA for this trial
                    # and drop the current TTA value from the list
                    # the same TTA will be used for next trials until there's another left turnx
                    if (current_turn==1):
                        tta = tta_values[-1]
                        tta_values = tta_values[:-1]

                    # distance to the center of the ego car
                    d_condition = random.choice(self.bot_distance_values)
                    bot_speed = d_condition/tta

                    is_turn_completed = False
                    is_at_active_intersection = False
                    is_first_cue_played = False
                    is_second_cue_played = False

                    intersection_coordinates = (self.active_intersection[0]*self.block_size,
                                                self.active_intersection[1]*self.block_size)

                    # whenever we exchange y-coordinates with Carla server, we invert the sign
                    active_intersection_loc = carla.Location(x=intersection_coordinates[0],
                                                             y=-intersection_coordinates[1],
                                                             z=0.0)
                    trial_log = []
                    trial_start_time = time.time()

                    print ('Current trial: %i, turn %f, TTA %f, bot speed %f, distance %f' %
                            (j+1, current_turn, tta, bot_speed, d_condition))

                    while not is_turn_completed:
                        t = time.time()-trial_start_time
                        speed = np.sqrt(self.ego_actor.get_velocity().x**2 + self.ego_actor.get_velocity().y**2)
                        ego_distance_to_intersection = self.ego_actor.get_location().distance(active_intersection_loc)
                        '''
                        'subj_id', 'session', 'route', 'intersection_no',
                        'intersection_x', 'intersection_y', 'turn_direction', 't',
                        'ego_distance_to_intersection', 'tta_condition', 'd_condition', 'v_condition'
                        '''
                        values_to_log = list(['%i' % value for value in
                                          [self.exp_info['subj_id'], self.exp_info['session'], route_number, j+1,
                                         intersection_coordinates[0], intersection_coordinates[1], current_turn] ]) \
                                      + list(['%.4f' % value for value in
                                          [t, ego_distance_to_intersection, tta, d_condition, bot_speed]])

                        self.update_log(trial_log, values_to_log)

                        self.update_ego_control()

                        if not self.bot_actor is None:
                            self.update_bot_control(bot_speed)

                        self.noise_sound.set_volume(0.05 + speed/20)

                        if((not is_first_cue_played) & (ego_distance_to_intersection<(4/5)*self.block_size)):
                            self.play_sound_cue(1, current_turn)
                            is_first_cue_played = True
                        # When the driver approaches the intersection, we play the second sound cue and destroy the bot at the previous intersection
                        elif((not is_second_cue_played) & (ego_distance_to_intersection<(1/5)*self.block_size)):
                            self.play_sound_cue(2, current_turn)
                            is_second_cue_played = True
                        elif((not is_at_active_intersection) & (ego_distance_to_intersection<10)):
                            is_at_active_intersection = True
                        # if at the left turn, wait until almost a full stop before spawning a bot
                        elif((current_turn==1) & (is_at_active_intersection) & (speed<1) &
                                                                        (self.bot_actor is None)):
                            self.spawn_bot(distance_to_intersection=d_condition-ego_distance_to_intersection,
                                           speed=bot_speed)
                        # if at the right turn, don't wait for slowdown when spawning a bot
                        elif((current_turn==-1) & (is_at_active_intersection) & (self.bot_actor is None)):
                            self.spawn_bot(75, bot_speed)
                        # When the driver leaves the intersection we designate the next intersection as active and destroy the bot
                        elif((is_at_active_intersection) & (ego_distance_to_intersection>10)):
                            print('updating origin and active intersection')
                            current_direction = self.active_intersection - self.origin
                            print(current_direction)
                            new_origin = self.active_intersection
                            print(new_origin)
                            new_active_intersection = (self.active_intersection +
                                            self.rotate(current_direction, np.pi/2*current_turn))
                            print(new_active_intersection)
                            self.origin = new_origin
                            self.active_intersection = new_active_intersection

                            if (not (self.bot_actor is None)):
                                self.bot_actor.destroy()
                                self.bot_actor = None

                            is_turn_completed = True

                        time.sleep(0.01)

                    self.write_log(trial_log)

                self.noise_sound.stop()

                if (not (self.ego_actor is None)):
                    self.ego_actor.destroy()
                    self.ego_actor = None
                if (not (self.bot_actor is None)):
                    self.bot_actor.destroy()
                    self.bot_actor = None
                self.joystick.quit()

                time.sleep(5.0)

        except KeyboardInterrupt:
            for actor in self.world.get_actors():
                actor.destroy()

def main():
    ltap_carla_client = LTAPCarlaClient()
    ltap_carla_client.run()

if __name__ == '__main__':
    main()


