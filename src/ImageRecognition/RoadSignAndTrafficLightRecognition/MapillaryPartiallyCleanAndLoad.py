#Alex Eliseev
#The second dataset used for image recognition was the S2TLD, this script adds the Mapillary partially annotated dataset into the training, testing and validation pool.
#With it, the recognition model is going to be able to recognize far more road signs.
#Script converts mapillary json annotation files into yolo format and saves them as txt and appends the txt alongside its respective image to the 
#pool of training, validation and testing images.
#Script exact same apart from file locations as the MapillaryCleanAndLoad
#After initial staderdization, the map reduce class was ran on the results to remove more of the "other sign" class in the seperate folder.
#imports
import os
import json
import shutil
from pathlib import Path
#Setting up sets, each class from the Mapillary Jsons is converted into one of mine, some generalizations are made to reduce class count
#I.E multiple Mapillary classes can be merged into one of mine. A dictionairy after is used to return my class number as can be found in the yaml file.
SetStopSigns = {
    "regulatory--all-way--g1","regulatory--stop--g10","regulatory--stop--g1","regulatory--stop--g2","regulatory--stop--g3","regulatory--stop--g4","regulatory--stop--g5",
    "regulatory--stop--g6","regulatory--stop--g7","regulatory--stop--g8","regulatory--stop--g9"
}
SetStopAhead = {
    "warning--stop-ahead--g1","warning--stop-ahead--g3","warning--stop-ahead--g4","warning--stop-ahead--g5","warning--stop-ahead--g6"
}
SetTrafficLightAhead = {
    "warning--traffic-signals--g1","warning--traffic-signals--g2","warning--traffic-signals--g3","warning--traffic-signals--g4","warning--traffic-signals--g5","warning--traffic-signals--g6"
}
SetCarsBannedFromLane = {
    "regulatory--bicycles-and-buses-only--g1","regulatory--bicycles-only--g1","regulatory--bicycles-only--g2","regulatory--bicycles-only--g3",
    "regulatory--bicycles-only--g4","regulatory--bus-priority-lane--g1","regulatory--buses-and-taxi-only--g1","regulatory--buses-only--g1",
    "regulatory--buses-only--g2","regulatory--light-rail-only--g1","regulatory--mopeds-and-bicycles-only--g1","regulatory--motorcycles-and-bicycles-only--g1","regulatory--motorcycles-only--g1",
    "regulatory--motorcycles-only--g2","regulatory--taxi-only--g1","regulatory--trams-and-buses-only--g1","regulatory--trams-only--g1","regulatory--trucks-only--g1"
}
SetNonCarsBannedFromLane = {
    "regulatory--end-of-bicycles-only--g1","regulatory--end-of-bicycles-only--g2","regulatory--end-of-bus-and-taxi-only--g1",
    "regulatory--end-of-buses-only--g1","regulatory--end-of-buses-only--g2","regulatory--end-of-trams-and-buses-only--g1",
    "regulatory--end-of-trams-only--g1","regulatory--end-of-trucks-and-buses-only--g1","regulatory--end-of-trucks-only--g1","regulatory--end-of-trucks-only--g2","regulatory--no-heavy-goods-vehicles--g1",
    "regulatory--no-heavy-goods-vehicles--g2","regulatory--no-heavy-goods-vehicles--g3","regulatory--no-heavy-goods-vehicles--g4","regulatory--no-heavy-goods-vehicles--g5","regulatory--no-buses--g1",
    "regulatory--no-buses--g2","regulatory--no-buses--g3"
}
SetDeadEnd = {
    "information--dead-end--g1","information--dead-end--g2","information--dead-end--g3","information--dead-end--g4","warning--dead-end--g1","warning--dead-end--g2","warning--dead-end--g3",
    "warning--dead-end-go-left--g1","warning--dead-end-go-right--g1","complementary--dead-end--g1"
}
SetNoEntry = {
    "regulatory--no-entry--g1"
}
SetRoadClosed = {
    "regulatory--road-closed--g1","regulatory--road-closed--g2","regulatory--road-closed-to-vehicles--g1","regulatory--road-closed-to-vehicles--g3","warning--ramp-closed--g1","warning--road-blocks--g1",
    "warning--road-closed--g3"
}
SetYield = {
    "regulatory--give-way-to-oncoming-traffic--g1","regulatory--give-way-to-oncoming-traffic--g2","regulatory--yield--g1","regulatory--yield-or-stop-for-pedestrians--g1"
}
SetYieldAhead = {
    "warning--yield-ahead--g1", "warning--yield-ahead--g3","complementary--chevron-right--g3","complementary--chevron-right--g4","complementary--chevron-right--g5"
}
SetYieldPedestrians = {
    "regulatory--in-street-pedestrian-crossing--g1","information--pedestrians-crossing--g1","information--pedestrians-crossing--g2","information--pedestrians-crossing--g3",
    "information--safety-zone--g1","information--safety-zone--g2","information--safety-zone--g3","warning--disabled-persons-crossing--g1","warning--disabled-persons-crossing--g2",
    "warning--pedestrians-crossing--g10","warning--pedestrians-crossing--g11","warning--pedestrians-crossing--g12","warning--pedestrians-crossing--g1","warning--pedestrians-crossing--g4","warning--pedestrians-crossing--g5",
    "warning--pedestrians-crossing--g6","warning--pedestrians-crossing--g7","warning--pedestrians-crossing--g8","warning--pedestrians-crossing--g9","warning--skiers--g1","warning--skiers--g2",
    "warning--skiers--g3","warning--trail-crossing--g1","warning--trail-crossing--g2","warning--trail-crossing--g3","warning--trail-crossing--g4","warning--trail-crossing--g5","warning--trail-crossing--g6","complementary--disabled-persons--g1"
}
SetYieldBicycles = {
    "regulatory--give-way-to-bicycles--g1","warning--bicycles-and-others--g1","warning--bicycles-crossing--g1","warning--bicycles-crossing--g2","warning--bicycles-crossing--g3","information--bicycles-crossing--g1","information--bicycles-crossing--g2","information--bicycles-crossing--g3",
    "warning--slippery-bicycles--g1","complementary--bicycles--g1"
}
SetRailRoadCrossing = {
    "warning--railroad-crossing--g1","warning--railroad-crossing--g2","warning--railroad-crossing--g3","warning--railroad-crossing--g4","warning--railroad-crossing-with-barriers--g1",
    "warning--railroad-crossing-with-barriers--g2","warning--railroad-crossing-with-barriers--g3","warning--railroad-crossing-with-barriers--g4","warning--railroad-crossing-with-barriers--g5",
    "warning--railroad-crossing-with-barriers--g6","warning--railroad-crossing-with-barriers--g7","warning--railroad-crossing-without-barriers--g1","warning--railroad-crossing-without-barriers--g2",
    "warning--railroad-crossing-without-barriers--g3","warning--railroad-crossing-without-barriers--g4","warning--railroad-crossing-without-barriers--g5","warning--railroad-crossing-without-barriers--g6",
    "warning--railroad-intersection--g1","warning--railroad-intersection--g2","warning--railroad-intersection--g3","warning--railroad-intersection--g4","warning--railroad-intersection--g5","warning--railroad-intersection--g6",
    "warning--railroad-intersection--g7","warning--railroad-intersection--g8","warning--railroad-intersection--g9"
}
SetCarsCantGoHere = {
    "regulatory--dual-path-bicycles-and-pedestrians--g1","regulatory--dual-path-bicycles-and-pedestrians--g2","regulatory--dual-path-bicycles-and-pedestrians--g3",
    "regulatory--dual-path-equestrians-and-pedestrians--g1","regulatory--dual-path-equestrians-and-pedestrians-bicycles--g1","regulatory--dual-path-pedestrians-and-bicycles--g1",
    "regulatory--dual-path-pedestrians-and-bicycles--g2","regulatory--dual-path-pedestrians-and-equestrians--g1","regulatory--dual-path-pedestrians-bicycles-and-equestrians--g1",
    "regulatory--no-motor-vehicles--g1","regulatory--no-motor-vehicles--g3","regulatory--no-motor-vehicles--g4","regulatory--no-motor-vehicles--g5","regulatory--no-motor-vehicles--g6",
    "regulatory--no-motor-vehicles--g7","regulatory--no-motor-vehicles-except-motorcycles--g1","regulatory--no-motor-vehicles-except-motorcycles--g2","regulatory--no-motor-vehicles-except-motorcycles--g3",
    "regulatory--no-motor-vehicles-or-bicycles--g1","regulatory--no-motor-vehicles-or-buses--g1","regulatory--no-motor-vehicles-or-carts--g1"

}
SetRoundaboutIntersection = {
    "regulatory--circular-intersection--g1","regulatory--circular-intersection--g2","regulatory--circular-intersection--g3",
    "regulatory--circular-intersection--g4","regulatory--roundabout--g1","regulatory--roundabout--g2","regulatory--roundabout--g3","warning--roundabout--g1","warning--roundabout--g2",
    "warning--roundabout--g3","warning--roundabout--g4","warning--roundabout--g5","warning--roundabout--g6","warning--roundabout--g7"
}
SetDetour = {
    "regulatory--detour-left--g1","regulatory--detour-right--g1","complementary--detour--g1"
}
SetLaneMergesAhead = {
    "information--traffic-merges-left--g1","information--traffic-merges-right--g1","warning--closed-lane-in-triple-lanes--g1","warning--closed-lane-in-triple-lanes--g2","warning--entering-roadway-merge--g1",
    "warning--entering-roadway-merge--g2","warning--lane-closed-in-dual-lanes-left--g1","warning--lane-closed-in-dual-lanes-left--g2","warning--lane-closed-in-dual-lanes-right--g1","warning--lane-closed-in-dual-lanes-right--g2",
    "warning--occupied-lanes--g1","warning--road-narrows--g1","warning--road-narrows--g2","warning--road-narrows-left--g1","warning--road-narrows-left--g2","warning--road-narrows-left-ahead--g1",
    "warning--road-narrows-right--g1","warning--road-narrows-right--g2","warning--road-narrows-right-ahead--g1","warning--traffic-merges-left--g1","warning--traffic-merges-left--g2",
    "warning--traffic-merges-left--g3","warning--traffic-merges-left--g4","warning--traffic-merges-left-and-right--g1","warning--traffic-merges-left-buses--g1","warning--traffic-merges-right--g1","warning--traffic-merges-right--g2",
    "warning--traffic-merges-right--g3","warning--traffic-merges-right-buses--g1"
}
SetWarningMild = {
    "regulatory--divided-highway-crossing--g1","regulatory--divided-highway-ends--g1","regulatory--divided-highway-starts--g1","regulatory--low-speed-vehicle-permitted--g1",
    "warning--atv-crossing--g1","warning--atv-crossing--g2","warning--bear-crossing--g1","warning--bear-crossing--g2","warning--camel-crossing--g1","warning--camel-crossing--g2","warning--cliff--g1",
    "warning--cliff--g2","warning--crossroads--g1","warning--crossroads--g2","warning--crossroads--g3","warning--crossroads--g4","warning--crossroads--g5","warning--crossroads--g6","warning--crossroads-with-priority-to-the-right--g1",
    "warning--curve-left--g1","warning--curve-left--g2","warning--curve-left--g3","warning--curve-left-with-junction--g1","warning--curve-out-intersection-left--g1","warning--curve-out-intersection-right--g1",
    "warning--curve-right--g1","warning--curve-right--g2","warning--curve-right--g3","warning--curve-right-with-junction--g1","warning--dangerous-crosswinds-left--g1","warning--dangerous-crosswinds-left--g2",
    "warning--dangerous-crosswinds-left--g4","warning--dangerous-crosswinds-right--g1","warning--dangerous-crosswinds-right--g2","warning--dangerous-crosswinds-right--g3","warning--dangerous-crosswinds-right--g4",
    "warning--detour-ahead--g1","warning--detour-or-construction-ahead--g1","warning--divided-highway--g1","warning--divided-highway--g2","warning--divided-highway--g3","warning--divided-highway--g4",
    "warning--divided-highway--g5","warning--divided-highway--g6","warning--divided-highway--g7","warning--divided-highway--g8","warning--divided-highway--g9","warning--divided-highway-ends--g1",
    "warning--divided-highway-ends--g2","warning--divided-highway-ends--g3","warning--divided-highway-ends--g4","warning--divided-highway-on-left--g1","warning--divided-highway-on-left--g2","warning--divided-highway-on-right--g1",
    "warning--divided-highway-on-right--g2","warning--divided-highway-to-left--g1","warning--divided-highway-to-right--g1","warning--domestic-animals--g1","warning--domestic-animals--g2","warning--domestic-animals--g3",
    "warning--domestic-animals--g4","warning--domestic-animals--g5","warning--domestic-animals--g6","warning--domestic-animals--g7","warning--domestic-animals--g8","warning--double-side-roads-left--g1",
    "warning--double-side-roads-left--g3","warning--double-side-roads-right--g1","warning--double-side-roads-right--g3","warning--double-turn-first-left--g1","warning--double-turn-first-right--g1","warning--elephant-crossing--g1",
    "warning--emergency-vehicles--g1","warning--emu-crossing--g1","warning--emu-crossing--g2","warning--equestrians-crossing--g1","warning--equestrians-crossing--g2","warning--falling-rocks-or-debris-left--g1",
    "warning--falling-rocks-or-debris-left--g2","warning--falling-rocks-or-debris-left--g3","warning--falling-rocks-or-debris-left--g4","warning--falling-rocks-or-debris-right--g1","warning--falling-rocks-or-debris-right--g2",
    "warning--falling-rocks-or-debris-right--g3","warning--falling-rocks-or-debris-right--g4","warning--foggy-road--g1","warning--foggy-road--g2","warning--forest--g1","warning--frog-crossing--g1",
    "warning--gate--g1","warning--gate-left--g1","warning--gate-right--g1","warning--golf-carts-crossing--g1","warning--horse-crossing--g1","warning--icy-road--g1","warning--kangaroo-crossing--g1","warning--kiwi-crossing--g1",
    "warning--kiwi-crossing--g2","warning--koala-crossing--g1","warning--koala-crossing--g2","warning--koala-crossing--g3","warning--koala-crossing--g4","warning--limited-lighting-under-trees--g1","warning--logging-vehicles--g1",
    "warning--monkey-crossing--g1","warning--motorcycles-crossing--g1","warning--other-danger--g1","warning--other-danger--g2","warning--other-danger--g3","warning--opening-or-swing-bridge--g1","warning--opening-or-swing-bridge--g2",
    "warning--panda-crossing--g1","warning--polar-bear-crossing--g1","warning--quay-or-river-bank--g1","warning--quay-or-river-bank--g2","warning--quay-or-river-bank--g3","warning--quay-or-river-bank--g4","warning--rabbit-crossing--g1",
    "warning--raccoon-crossing--g1","warning--severe-weather--g1","warning--slippery-road-surface--g1","warning--slippery-road-surface--g2","warning--snow-tractors--g1","warning--snowmobiles--g1","warning--snowmobiles--g2",
    "warning--snowmobiles--g3","warning--snowmobiles-and-others--g1","warning--soft-shoulder--g1","warning--soft-shoulder--g2","warning--soft-shoulder--g3","warning--soft-shoulder--g4","warning--speed-camera--g1","warning--steep-ascent--g1","warning--steep-ascent--g2",
    "warning--steep-ascent--g3","warning--steep-ascent--g4","warning--steep-ascent-and-descent--g1","warning--tanks-crossing--g1","warning--tanks-crossing--g2","warning--texts--g1","warning--texts--g3","warning--texts--g2",
    "warning--towing--g1","warning--tractors--g1","warning--tractors--g2","warning--tractors--g3","warning--tractors--g4","warning--tractors--g5","warning--tractors--g6","warning--tractors--g7","warning--traffic-queues-likely--g1","warning--traffic-queues-likely--g2",
    "warning--traffic-queues-likely--g3","warning--traffic-queues-likely--g4","warning--traffic-queues-likely--g5","warning--trams-crossing--g1","warning--trams-crossing--g2","warning--trucks-crossing--g1",
    "warning--trucks-crossing--g2","warning--trucks-rollover--g1","warning--trucks-rollover--g2","warning--trucks-rollover--g3","warning--trucks-rollover--g4","warning--trucks-rollover--g5","warning--tunnel--g1",
    "warning--tunnel--g2","warning--tunnel--g3","warning--tunnel--g4","warning--tunnel--g5","warning--tunnel--g6","warning--tunnel--g7","warning--two-way-traffic--g1","warning--two-way-traffic--g2",
    "warning--two-way-traffic--g3","warning--two-way-traffic--g4","warning--two-way-traffic--g5","warning--two-way-traffic--g6","warning--vehicles-crossing--g1","warning--wild-animals--g1","warning--wild-animals--g2",
    "warning--wild-animals--g3","warning--wild-animals--g4","warning--wild-animals--g5","warning--wild-animals--g6","warning--wild-animals--g7","warning--wild-animals--g8","warning--wind--g1","warning--wombat-crossing--g1"
}
SetIntersectionAhead = {
    "warning--junction-with-a-side-road-acute-left--g1","warning--junction-with-a-side-road-acute-left--g2","warning--junction-with-a-side-road-acute-right--g1","warning--junction-with-a-side-road-acute-right--g2","warning--junction-with-a-side-road-perpendicular-left--g1",
    "warning--junction-with-a-side-road-perpendicular-left--g2","warning--junction-with-a-side-road-perpendicular-left--g3","warning--junction-with-a-side-road-perpendicular-left--g4","warning--junction-with-a-side-road-perpendicular-right--g1",
    "warning--junction-with-a-side-road-perpendicular-right--g2","warning--junction-with-a-side-road-perpendicular-right--g3","warning--junction-with-a-side-road-perpendicular-right--g4","warning--junction-with-merge-from-left--g1",
    "warning--junction-with-merge-from-right--g1","warning--junction-with-side-roads--g1","warning--offset-roads--g1","warning--offset-roads--g2","warning--offset-roads--g3","warning--offset-roads--g4","warning--skewed-t-roads-left--g1",
    "warning--skewed-t-roads-left--g2","warning--skewed-t-roads-left--g3","warning--skewed-t-roads-right--g1","warning--skewed-t-roads-right--g2","warning--skewed-t-roads-right--g3","warning--t-roads--g1",
    "warning--t-roads--g2","warning--y-roads--g1","warning--y-roads--g2"
}
SetWarningDescentSlowDown = {
    "warning--steep-descent--g1","warning--steep-descent--g2","warning--steep-descent--g3","warning--steep-descent--g4",
    "warning--steep-descent--g5","warning--steep-descent--g6","warning--double-descent--g1"
}
SetWarningConsiderate = {
    "warning--accident-area--g1","warning--accident-area--g2","warning--accident-area--g3","warning--accident-area--g4","warning--accident-area--g5","warning--accident-area--g6",
    "warning--accident-area--g7","warning--accident-area--g8","warning--fresh-oil--g1","warning--slow--g1","warning--traffic-slow--g1","warning--uneven-road--g1","warning--uneven-road--g2",
    "warning--uneven-roads-ahead--g1","complementary--accident-area--g1","complementary--accident-area--g2","complementary--accident-area--g3","complementary--accident-area--g4"

}
SetWindingRoad = {
    "warning--winding-road--g1","warning--winding-road-first-left--g1","warning--winding-road-first-left--g2","warning--winding-road-first-left--g3","warning--winding-road-first-right--g1",
    "warning--winding-road-first-right--g2","warning--winding-road-first-right--g3","warning--winding-road-first-right--g4","warning--winding-road-to-left--g1","warning--winding-road-to-right--g1"
}
SetWarningSevere = {
    "warning--hairpin-curve-left--g1","warning--double-reverse-curve-left--g1","warning--double-reverse-curve-left--g2","warning--double-reverse-curve-right--g1","warning--double-reverse-curve-right--g2",
    "warning--double-curve-first-right--g2","warning--double-curve-first-left--g1","warning--hairpin-curve-left--g2","warning--hairpin-curve-left--g3","warning--hairpin-curve-right--g1","warning--loop-270-degree--g1","warning--loop-pretzel--g1",
    "warning--hairpin-curve-right--g3","warning--sharp-turn--g1","warning--single-reverse-curve--g1","warning--triple-curve-left--g1","warning--triple-curve-right--g1","warning--triple-reverse-curve-left--g1",
    "warning--triple-reverse-curve-right--g1"
}
SetWarningConstruction = {
    "warning--roadworks--g10","warning--construction-ahead--g1","warning--flaggers-in-road--g1","warning--flaggers-in-road--g2","warning--roadworks--g11","warning--roadworks--g1","warning--roadworks--g2",
    "warning--roadworks--g3","warning--roadworks--g5","warning--roadworks--g6","warning--roadworks--g8","warning--roadworks--g9","warning--roadworks-go-left-or-straight--g1","warning--roadworks-go-right-or-straight--g1",

}
SetNoPassing = {
    "regulatory--do-not-pass--g1","regulatory--light-rail-do-not-pass--g1","regulatory--no-overtaking--g1","regulatory--no-overtaking--g2","regulatory--no-overtaking--g4","regulatory--no-overtaking--g5",
    "regulatory--no-overtaking--g6","regulatory--no-overtaking--g7","warning--double-curve-first-left--g2","warning--double-curve-first-right--g1","warning--no-passing-zone--g1","warning--no-passing-zone--g2",

}
SetPassIslandOnEitherSideAhead = {
    "regulatory--pass-on-either-side--g1","regulatory--pass-on-either-side--g2","regulatory--pass-on-either-side--g3","information--pass-on-either-side--g1","warning--pass-left-or-right--g1",
    "warning--pass-left-or-right--g2"
}
SetPassingAllowed = {
    "regulatory--end-of-no-overtaking--g1","regulatory--end-of-no-overtaking--g2","regulatory--end-of-no-overtaking--g3","regulatory--end-of-no-overtaking--g4","regulatory--end-of-no-overtaking--g5",
    "regulatory--pass-with-care--g1"
    

}
SetParkingAllowed = {
    "regulatory--end-of-no-parking--g1","regulatory--end-of-no-parking--g2","regulatory--end-of-no-parking-or-stopping--g1","information--parallel-parking--g1","information--park-and-ride--g1",
    "information--park-and-ride--g2","information--parking--g1","information--parking--g2","information--parking--g3","information--parking--g4","information--parking--g5","information--parking--g6",
    "information--parking-area--g1","information--passenger-loading-zone--g1"
}
SetNoParking = {
    "regulatory--end-of-parking-zone--g1","regulatory--end-of-parking-zone--g2","regulatory--no-parking--g1","regulatory--no-parking--g2","regulatory--no-parking--g3",
    "regulatory--no-parking--g4","regulatory--no-parking--g5","regulatory--no-parking--g6","regulatory--no-parking--g7","regulatory--no-parking--g8","regulatory--no-parking--g9","regulatory--no-parking-bus-stop--g1",
    "regulatory--no-parking-or-no-stopping--g1","regulatory--no-parking-or-no-stopping--g2","regulatory--no-parking-or-no-stopping--g3","regulatory--no-parking-or-no-stopping--g4",
    "regulatory--no-parking-or-no-stopping--g5","regulatory--no-passenger-loading--g1"
}
SetReservedParking = {
    "information--disabled-persons--g1","information--disabled-persons--g2","information--disabled-persons--g3"
}
SetNoStopping = {
    "regulatory--no-stopping--g1","regulatory--no-stopping--g2","regulatory--no-stopping--g3","regulatory--no-stopping--g5","regulatory--no-stopping--g6","regulatory--no-stopping--g7",
    "regulatory--no-stopping-on-pavement--g1"
}
SetHonkingDisabled = {
    "regulatory--horn--g1","regulatory--no-horn--g1","regulatory--no-horn--g2"
}
SetTurnAnyDirection = {
    "regulatory--end-of-one-way-straight--g1"
}
SetOnlyGoStraight = {
    "regulatory--go-straight--g1","regulatory--go-straight--g3","regulatory--one-way-straight--g3","regulatory--one-way-straight--g1","information--go-straight--g1"
}
SetLeftLeftMiddleStraightRightRight = {
    "warning--triple-lanes-with-directions--g1"
}
SetChevronLeft = {
    "complementary--chevron-left--g1","complementary--chevron-left--g2","complementary--chevron-left--g3","complementary--chevron-left--g4","complementary--chevron-left--g5"
}
SetChevronRight = {
    "complementary--chevron-right--g1","complementary--chevron-right--g2","complementary--chevron-right--g3","complementary--chevron-right--g4","complementary--chevron-right--g5"
}
SetTurnLeft = {
    "regulatory--dual-lanes-turn-left--g1","regulatory--dual-lanes-turn-left-no-u-turn--g1","regulatory--keep-left--g1","regulatory--keep-left--g2","regulatory--keep-left--g3",
    "regulatory--keep-left--g4","regulatory--keep-left--g5","regulatory--keep-left--g6","regulatory--keep-left--g7","regulatory--one-way-left--g1","regulatory--one-way-left--g2",
    "regulatory--one-way-left--g3","regulatory--turn-left--g1","regulatory--turn-left--g2","regulatory--turn-left--g3","regulatory--turn-left-ahead--g1","regulatory--turn-left-ahead--g2",
    "regulatory--turn-left-or-u-turn--g1","information--go-left--g1","warning--dual-lanes-all-directions-on-left--g1","warning--dual-lanes-left-turn--g1","warning--dual-lanes-turn-left--g1",
    "warning--go-left--g1","warning--keep-left--g1","warning--keep-right--g1","warning--triple-lanes-left-turn--g1","warning--turn-left--g1","warning--turn-left--g2","warning--turn-left--g3",
}
SetTurnRight = {
    "regulatory--keep-right--g1","regulatory--keep-right--g2","regulatory--keep-right--g3","regulatory--keep-right--g4","regulatory--keep-right--g5","regulatory--keep-right--g6",
    "regulatory--keep-right--g7","regulatory--keep-right--g8","regulatory--keep-right--g9","regulatory--one-way-right--g1","regulatory--one-way-right--g2","regulatory--one-way-right--g3",
    "regulatory--turn-right--g1","regulatory--turn-right--g2","regulatory--turn-right--g3","regulatory--turn-right-ahead--g1","regulatory--turn-right-ahead--g2","information--go-right--g1",
    "warning--dual-lanes-all-directions-on-right--g1","warning--dual-lanes-right-turn--g1","warning--dual-lanes-turn-right--g1","warning--go-right--g1","warning--triple-lanes-right-turn--g1",
    "warning--turn-right--g1","warning--turn-right--g2","warning--turn-right--g3"
}
SetTurnLeftOrRight = {
    "regulatory--turn-left-or-right--g1","regulatory--turn-left-or-right--g2","regulatory--turn-left-or-right--g3","warning--turn-left-or-right--g1"
}
SetLeftLeftRightRight = {
    "warning--dual-lanes-turn-left-or-right--g2","warning--dual-lanes-turn-left-or-right--g1","warning--dual-lanes-turn-left-or-right--g3","warning--dual-lanes-turn-left-or-right--g4"
}
SetTurnLeftOrStraight = {
    "regulatory--go-straight-or-turn-left--g3","information--go-straight-or-left--g1","information--go-straight-or-turn-left--g1","regulatory--go-straight-or-turn-left--g2","regulatory--go-straight-or-turn-left--g1",
    "complementary--go-straight-or-turn-left--g1"
}
SetTurnRightOrStraight = {
    "regulatory--go-straight-or-turn-right--g1","regulatory--go-straight-or-turn-right--g2","regulatory--go-straight-or-turn-right--g3",
    "information--go-straight-or-right--g1","information--go-straight-or-turn-right--g1","complementary--go-straight-or-turn-right--g1"
}
SetStayLeftToGoStraight = {
    "regulatory--dual-lanes-go-straight-on-left--g1","warning--dual-lanes-right-turn-or-go-straight--g1","regulatory--dual-lanes-turn-right-or-straight--g1","warning--dual-lanes-go-straight-or-turn-right--g1",
    "warning--triple-lanes-right-turn-or-go-straight--g1"
}
SetStayRightToGoStraight = {
    "regulatory--dual-lanes-go-straight-on-right--g1","warning--dual-lanes-left-turn-or-go-straight--g1","warning--dual-lanes-go-straight-or-turn-left--g1","regulatory--dual-lanes-turn-left-or-straight--g1",
    "warning--triple-lanes-left-turn-or-go-straight--g1"
}
SetNoLeftTurn = {
    "regulatory--no-left-or-u-turn--g1","regulatory--no-left-turn--g1","regulatory--no-left-turn--g2","regulatory--no-left-turn--g3","regulatory--no-left-turn--g4","regulatory--no-left-turn--g5",

}
SetNoRightTurn = {
    "regulatory--no-right-turn--g1","regulatory--no-right-turn--g2","regulatory--no-right-turn--g3"
}
SetNoRightTurnOnRed = {
    "regulatory--no-right-turn-on-red--g1","regulatory--no-turn-on-red--g1","regulatory--no-turn-on-red--g2","regulatory--no-turn-on-red--g3"
}
SetNoLeftTurnOrStraight = {
    "regulatory--no-go-straight-or-turn-left--g1"
}
SetNoRightTurnOrStraight = {
    "regulatory--no-go-straight-or-turn-right--g1"
}
SetNoTurns = {
    "regulatory--no-turns--g1","regulatory--no-turns--g2","regulatory--no-horizontal-turn--g1"
}
SetNoUTurn = {
    "regulatory--no-u-turn--g1","regulatory--no-u-turn--g2","regulatory--no-u-turn--g3"
}
SetUTurnAllowed = {
    "regulatory--u-turn--g1","regulatory--u-turn--g2","regulatory--u-turn--g3","warning--u-turn--g1","warning--u-turn--g2"
}
SetDualSpeedLimit = {
    "regulatory--dual-speed-limits--g1","regulatory--dual-speed-limits--g2"
}
SetSpeedLimit100 = {
    "regulatory--maximum-speed-limit-100--g1","regulatory--maximum-speed-limit-100--g3","regulatory--maximum-speed-limit-led-100--g1"
}
SetSpeedLimit110 = {
    "regulatory--maximum-speed-limit-110--g1","regulatory--maximum-speed-limit-110--g3","regulatory--maximum-speed-limit-led-110--g1"
}
SetSpeedLimit120 = {
    "regulatory--maximum-speed-limit-120--g1","regulatory--maximum-speed-limit-120--g3","regulatory--maximum-speed-limit-led-120--g1"
}
SetSpeedLimit130 = {
    "regulatory--maximum-speed-limit-130--g1","regulatory--maximum-speed-limit-130--g3","regulatory--maximum-speed-limit-led-130--g1"
}
SetSpeedLimit20 = {
    "regulatory--maximum-speed-limit-20--g1","regulatory--maximum-speed-limit-20--g3","regulatory--maximum-speed-limit-led-20--g1","regulatory--maximum-speed-limit-led-20--g2","regulatory--maximum-speed-limit-led-20--g3"
}
SetSpeedLimit30 = {
    "regulatory--maximum-speed-limit-30--g1","regulatory--maximum-speed-limit-30--g3","regulatory--maximum-speed-limit-led-30--g1","regulatory--maximum-speed-limit-led-30--g2","regulatory--maximum-speed-limit-led-30--g3"
}
SetSpeedLimit40 = {
    "regulatory--maximum-speed-limit-40--g1","regulatory--maximum-speed-limit-40--g3","regulatory--maximum-speed-limit-led-40--g1","regulatory--maximum-speed-limit-led-40--g2","regulatory--maximum-speed-limit-led-40--g3"
}
SetSpeedLimit50 = {
    "regulatory--maximum-speed-limit-50--g1","regulatory--maximum-speed-limit-50--g3","regulatory--maximum-speed-limit-led-50--g1","regulatory--maximum-speed-limit-led-50--g2","regulatory--maximum-speed-limit-led-50--g3"
}
SetSpeedLimit60 = {
    "regulatory--maximum-speed-limit-60--g1","regulatory--maximum-speed-limit-60--g3","regulatory--maximum-speed-limit-led-60--g1","regulatory--maximum-speed-limit-led-60--g2","regulatory--maximum-speed-limit-led-60--g3"
}
SetSpeedLimit70 = {
    "regulatory--maximum-speed-limit-70--g1","regulatory--maximum-speed-limit-70--g3","regulatory--maximum-speed-limit-led-70--g1","regulatory--maximum-speed-limit-led-70--g2","regulatory--maximum-speed-limit-led-70--g3"
}
SetSpeedLimit80 = {
    "regulatory--maximum-speed-limit-80--g1","regulatory--maximum-speed-limit-80--g3","regulatory--maximum-speed-limit-led-80--g1","regulatory--maximum-speed-limit-led-80--g2","regulatory--maximum-speed-limit-led-80--g3"
}
SetSpeedLimit90 = {
    "regulatory--maximum-speed-limit-90--g1","regulatory--maximum-speed-limit-90--g3","regulatory--maximum-speed-limit-led-90--g1"
}
SetEndOfTemporarySpeedLimit = {
    "regulatory--end-of-maximum-speed-limit--g1","regulatory--end-of-maximum-speed-limit-10--g1","regulatory--end-of-maximum-speed-limit-10--g2",
    "regulatory--end-of-maximum-speed-limit-100--g1","regulatory--end-of-maximum-speed-limit-100--g2","regulatory--end-of-maximum-speed-limit-110--g1",
    "regulatory--end-of-maximum-speed-limit-110--g2","regulatory--end-of-maximum-speed-limit-120--g1","regulatory--end-of-maximum-speed-limit-120--g2",
    "regulatory--end-of-maximum-speed-limit-130--g1","regulatory--end-of-maximum-speed-limit-130--g2","regulatory--end-of-maximum-speed-limit-20--g1",
    "regulatory--end-of-maximum-speed-limit-20--g2","regulatory--end-of-maximum-speed-limit-25--g1","regulatory--end-of-maximum-speed-limit-25--g2",
    "regulatory--end-of-maximum-speed-limit-30--g1","regulatory--end-of-maximum-speed-limit-30--g2","regulatory--end-of-maximum-speed-limit-35--g1",
    "regulatory--end-of-maximum-speed-limit-35--g2","regulatory--end-of-maximum-speed-limit-40--g1","regulatory--end-of-maximum-speed-limit-40--g2",
    "regulatory--end-of-maximum-speed-limit-50--g1","regulatory--end-of-maximum-speed-limit-50--g2","regulatory--end-of-maximum-speed-limit-60--g1",
    "regulatory--end-of-maximum-speed-limit-60--g2","regulatory--end-of-maximum-speed-limit-65--g1","regulatory--end-of-maximum-speed-limit-65--g2",
    "regulatory--end-of-maximum-speed-limit-70--g1","regulatory--end-of-maximum-speed-limit-70--g2","regulatory--end-of-maximum-speed-limit-75--g1",
    "regulatory--end-of-maximum-speed-limit-75--g2","regulatory--end-of-maximum-speed-limit-80--g1","regulatory--end-of-maximum-speed-limit-80--g2",
    "regulatory--end-of-maximum-speed-limit-90--g1","regulatory--end-of-maximum-speed-limit-90--g2","regulatory--end-of-speed-limit-zone--g1","regulatory--end-of-speed-limit-zone--g2",
    "regulatory--end-of-speed-limit-zone--g3",
}
SetSchoolZone = {
    "information--children--g1","information--children--g2","information--children-crossing--g5","warning--children--g1","warning--children--g2","warning--children--g3",
    "warning--children--g4","warning--children--g6","warning--playground--g1","warning--playground--g3","warning--school-zone--g2"
}
SetEndOfSchoolZone = {
    "regulatory--end-of-school-zone--g1"
}
SetClearance = {
    "regulatory--height-limit--g1","information--height-limit--g1","information--height-limit--g2","warning--height-restriction--g2","warning--height-restriction--g3","warning--height-restriction--g4",
    "warning--height-restriction--g5","warning--low-ground-clearance--g1","warning--low-ground-clearance--g2","warning--low-ground-clearance--g3","complementary--height-limit--g1",
    "complementary--height-limit--g2"
}
SetNoHighBeams = {
   "regulatory--low-beam-headlights--g1","regulatory--low-beam-headlights--g2","regulatory--low-beam-headlights--g3"
}
SetAirportSign = {
    "information--airport--g1","information--airport--g2","information--flight-port--g1"
}
SetHospitalSign = {
    "information--hospital--g1"
}
SetExit = {
    "information--exit-ahead--g1","information--exit-ahead--g2","information--exit-ahead--g3","information--highway-exit--g1","information--motorway-exit-ahead--g1",
    "information--motorway-exit-ahead--g2","information--motorway-exit-ahead--g3"
}
SetRoadBump = {
    "information--road-bump--g1","warning--pavement-ahead--g1","warning--road-bump--g1","warning--road-bump--g2","warning--road-bump--g3","warning--road-bump-with-speed-limit--g1","warning--ruts--g1"
}
SetRoadDip = {
    "warning--dip--g1","warning--dip--g2"
}
SetUnpavedRoad = {
    "warning--gravel-road-surface--g1","warning--loose-road-surface--g1","warning--loose-road-surface--g2","warning--loose-road-surface--g3","warning--loose-road-surface--g4","warning--pavement-ends--g1",
    "warning--pavement-ends--g2","warning--pavement-ends--g3","warning--pavement-ends--g4","warning--pavement-ends--g5","warning--sand--g1","warning--sand-drift--g1","warning--soft-road-surface--g1","warning--soft-road-surface--g2",

}
#Settting up a dictionairy of the sets and their Mapillary class names inside, and then mapping each class set to return the class number of my new classes
CLASS_MAPPING = {
    4: SetStopSigns,
    5: SetStopAhead,
    6: SetTrafficLightAhead,
    7: SetCarsBannedFromLane,
    8: SetNonCarsBannedFromLane,
    9: SetDeadEnd,
    10: SetNoEntry,
    11: SetRoadClosed,
    12: SetYield,
    13: SetYieldAhead,
    14: SetYieldPedestrians,
    15: SetYieldBicycles,
    16: SetRailRoadCrossing,
    17: SetCarsCantGoHere,
    18: SetRoundaboutIntersection,
    19: SetDetour,
    20: SetLaneMergesAhead,
    21: SetWarningMild,
    22: SetWarningConsiderate,
    23: SetWarningSevere,
    24: SetWarningDescentSlowDown,
    25: SetIntersectionAhead,
    26: SetWindingRoad,
    27: SetWarningConstruction,
    28: SetNoPassing,
    29: SetPassIslandOnEitherSideAhead,
    30: SetPassingAllowed,
    31: SetParkingAllowed,
    32: SetNoParking,
    33: SetReservedParking,
    34: SetNoStopping,
    35: SetHonkingDisabled,
    36: SetTurnAnyDirection,
    37: SetOnlyGoStraight,
    38: SetLeftLeftMiddleStraightRightRight,
    39: SetChevronLeft,
    40: SetChevronRight,
    41: SetTurnLeft,
    42: SetTurnRight,
    43: SetTurnLeftOrRight,
    44: SetLeftLeftRightRight,
    45: SetTurnLeftOrStraight,
    46: SetTurnRightOrStraight,
    47: SetStayLeftToGoStraight,
    48: SetStayRightToGoStraight,
    49: SetNoLeftTurn,
    50: SetNoRightTurn,
    51: SetNoRightTurnOnRed,
    52: SetNoLeftTurnOrStraight,
    53: SetNoRightTurnOrStraight,
    54: SetNoTurns,
    55: SetNoUTurn,
    56: SetUTurnAllowed,
    57: SetDualSpeedLimit,
    58: SetSpeedLimit100,
    59: SetSpeedLimit110,
    60: SetSpeedLimit120,
    61: SetSpeedLimit130,
    62: SetSpeedLimit20,
    63: SetSpeedLimit30,
    64: SetSpeedLimit40,
    65: SetSpeedLimit50,
    66: SetSpeedLimit60,
    67: SetSpeedLimit70,
    68: SetSpeedLimit80,
    69: SetSpeedLimit90,
    70: SetEndOfTemporarySpeedLimit,
    71: SetSchoolZone,
    72: SetEndOfSchoolZone,
    73: SetClearance,
    74: SetNoHighBeams,
    75: SetAirportSign,
    76: SetHospitalSign,
    77: SetExit,
    78: SetRoadBump,
    79: SetRoadDip,
    80: SetUnpavedRoad,
}

#Script opens Mapillary text file listing test, val or train JSONS
#Finds and opens JSOn from the shared JSONS folder
#Maps the JSON mapillary class labels unto mine and then saves them in YOLO format in a txt
#Finds the matching image based on the JSONS name in the mapillary images folder and then saves/appends the image and its new YOLO label to project's train/val/test folder
#in matching order
PROJECT_ROOT = Path(__file__).resolve().parents[3]
JSONS_MAPILLARY = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "raw" / "MapillaryPartiallyAnnotatedTrafficSignsDataset" / "mtsd_v2_partially_annotated" / "annotations"
IMAGES_MAPILLARY = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "raw" / "MapillaryPartiallyAnnotatedTrafficSignsDataset" / "images"

#First doing so for training images + labels from Mapillary
TRAIN_SPLIT_MAPILLARY_TXT = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "raw" / "MapillaryPartiallyAnnotatedTrafficSignsDataset" / "mtsd_v2_partially_annotated" / "splits" / "train.txt"
OUTPUT_LABEL_TRAIN_FOLDER = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "train2" / "labels" #Appending yolo labels to this folder
OUTPUT_IMAGE_TRAIN_FOLDER = PROJECT_ROOT / "Data" / "SignAndTrafficLights" / "train2" / "images" #Appending images for matching json/yolo labels to this folder

#Loop opens text file listing train JSONS and processes each one saving them + their class lables in YOLO format and converted to my classes as txt
#to a train/images+labels folders.
with open(TRAIN_SPLIT_MAPILLARY_TXT, "r") as f:
    for line in f:
        json_id = line.strip()

        if not json_id:
            continue
        
        #Opening the next listed JSON
        json_path = JSONS_MAPILLARY / f"{json_id}.json"

        if not json_path.exists():
            print(f"Missing: {json_path}")
            continue
        
        with open(json_path, "r") as jf:
            data = json.load(jf)
        
        json_name = json_path.stem #Getting JSON name
        output_txt_path = OUTPUT_LABEL_TRAIN_FOLDER / f"{json_name}.txt" #Txt location to which script is writing labels for image

        #Getting image dimensions from JSON
        img_width = data["width"]
        img_height = data["height"]
        yolo_lines = []

        #Going through each sign class in the JSON file, saving each to txt for the matching image
        for obj in data["objects"]:

            mapillary_label = obj["label"]

            #Default to Other_Sign
            class_id = 81

            #Finding matching class based on hand made mapping above
            for candidate_class_id, label_set in CLASS_MAPPING.items():
                if mapillary_label in label_set:
                    class_id = candidate_class_id
                    break

            bbox = obj["bbox"]

            xmin = bbox["xmin"]
            ymin = bbox["ymin"]
            xmax = bbox["xmax"]
            ymax = bbox["ymax"]

            #Converting to YOLO format
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            yolo_lines.append(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}"
            )

        #Saving yolo txt to the destination label folder by appending it there
        with open(output_txt_path, "w") as f:
            f.write("\n".join(yolo_lines))

        #Searching for image with matching name to the JSON in the image folder and also appending it to the destination folder
        #So label matches image
        source_image = IMAGES_MAPILLARY / f"{json_name}.jpg"
        if source_image.exists():

            destination_image = OUTPUT_IMAGE_TRAIN_FOLDER / source_image.name

            shutil.copy2(source_image, destination_image)
        else:
            print(f"Could not find image for {json_name}")



            