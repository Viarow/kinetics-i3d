
"""
Record the common classes between Kinetics400 and (UCF101, HMDB51, Moments In Time)
"""

Kinetics400_UCF101 = [('Archery', 5), ('BenchPress', 19), ('Biking', 22), ('BlowingCandles', 27),
                      ('Bowling', 31), ('BrushingTeeth', 37), ('RopeClimbing', 66), ('BabyCrawling', 77),
                      ('CliffDiving', 93), ('HammerThrow', 148), ('HighJump', 154), ('HulaHoop', 159),
                      ('JavelinThrow', 166), ('LongJump', 182), ('Lunges', 183), ('MoppingFloor', 198),
                      ('PlayingCello', 223), ('Drumming', 230), ('PlayingFlute',231), ('PlayingGuitar', 232),
                      ('PlayingPiano', 241), ('PlayingViolin', 250), ('PoleVault', 253), ('PullUps', 255),
                      ('PushUps', 260), ('Biking', 267), ('HorseRiding', 273), ('Shotput', 298),
                      ('SkateBoarding', 306), ('SkyDiving', 312), ('BreastStroke', 340), ('TaiChi', 346),
                      ('ThrowDiscus', 358), ('WalkingWithDog', 378)]

Kinetics400_HMDB51 = [('brush_hair', 36), ('cartwheel', 45), ('clap', 57), ('dribble', 99),
                      ('drink', 100), ('hug', 158), ('laugh', 180), ('punch', 258),
                      ('punch', 259), ('push_up', 260), ('ride_bike', 267), ('ride_horse', 273),
                      ('situp', 305)]

HMDB51_LEAK = [('kiss', 176), ('shake_hands', 288), ('smoke', 316)]