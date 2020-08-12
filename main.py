import app.src.core.travel_assist as ta
import app.src.core.image_homing_guidance as hg
import sys
import getopt

if __name__ == '__main__':

    try:
        if sys.argv[1] == 'travel-assist':
            if sys.argv[2] == 'ui':
                ui = ta.TravelUI()
                if sys.argv[3] == 'simulation':
                    ui.experiment('simulation')  # Either 'simulation', 'plot' or 'write text'

                elif sys.argv[3] == 'plot':
                    ui.experiment('plot')

                elif sys.argv[3] == 'write-text':
                    ui.experiment('write-text')
            elif sys.argv[2] == 'fast':
                sim = ta.Simulator()
                sim.simulate('app/datasets/travel-assist/sources/source-diverse/3.cloudy-images',
                             'app/datasets/travel-assist/sources/source-diverse/2.blurred', True, 5)
        elif sys.argv[1] == 'homing':
            cam_URL, camera_index, method, n_features, nn_dist, video, target = None, -1, 'ORB', 10000, 50, None, None
            try:
                opts, args = getopt.getopt(sys.argv[2:], 'c:i:m:n:d:v:t:')
            except IndexError:
                pass

            for opt, arg in opts:
                if opt in ['-c']:
                    cam_URL = arg
                elif opt in ['-i']:
                    camera_index = int(arg)
                elif opt in ['-m']:
                    method = arg
                elif opt in ['-n']:
                    n_features = int(arg)
                elif opt in ['-d']:
                    nn_dist = int(arg)
                elif opt in ['-v']:
                    video = arg
                elif opt in ['-t']:
                    target = arg

            hg.initialize_homing(cam_URL, camera_index, method, n_features, nn_dist, video, target)
    except IndexError as e:
        print(e)
