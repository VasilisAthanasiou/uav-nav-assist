import app.src.core.travel_assist as ta

import sys

if __name__ == '__main__':

    if sys.argv[1] == 'ui':
        ui = ta.TravelUI()
        if sys.argv[2] == 'simulation':
            ui.experiment('simulation')  # Either 'simulation', 'plot' or 'write text'

        elif sys.argv[2] == 'plot':
            ui.experiment('plot')

        elif sys.argv[2] == 'write-text':
            ui.experiment('write-text')
    elif sys.argv[1] == 'fast':
        sim = ta.Simulator()
        sim.simulate('app/datasets/travel-assist/sources/source-diverse/3.cloudy-images',
                     'app/datasets/travel-assist/sources/source-diverse/2.blurred', True, 5)
