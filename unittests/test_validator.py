import unittest
from helpers.resource_loader import ResourceLoader
from helpers.discover_testbeds import discover_testbeds
from optimization_loop import OptimizationLoop

class ValidatorTest(unittest.TestCase):
    def setUp(self):
        self.rl = ResourceLoader()
        # Dynamically discover all available testbed classes
        self.testbeds_map = discover_testbeds()

    def test_HoldTheLine(self):
        NoRobots = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        JCorrent = [
            303.8495134310999, 164.78849057293556, 113.97289336455995, 86.78895269391742,
            69.57203022873995, 58.08742655812385, 50.025536633363316, 43.901152937650316,
            39.19539902982932, 35.58899791864856
        ]
        deviation = 0.10  # 10%
        testbedName = "HoldTheLine"
        path = "../testbeds/{}/Parameters.properties".format(testbedName)

        # Instantiate the correct testbed class
        testbed_class = self.testbeds_map[testbedName]
        for i, nrob in enumerate(NoRobots):
            props = self.rl.get_properties_ap(path)
            props["randomID"] = "true"
            props["displayTime&CF"] = "false"
            props["noRobots"] = str(nrob)
            Tmax = int(props["noIter"])

            # Create a fresh instance of the testbed for each run
            testbed_instance = testbed_class()
            # Pass the testbed instance to OptimizationLoop
            opt_loop = OptimizationLoop(testbed_instance, propertiesFILE=props)

            curCF = opt_loop.getJJ()[Tmax - 1]
            print("[Number of robots: {}] Average CF value: {} | Recorded CF value: {}".format(
                nrob, JCorrent[i], curCF))
            self.assertTrue(
                JCorrent[i] * (1 - deviation) <= curCF <= JCorrent[i] * (1 + deviation)
            )

    def test_AdaptiveCoverage2D(self):
        testbedName = "AdaptiveCoverage2D"
        NoRobots = [5, 10, 15, 20]
        DesiredCF = [
            1964.9232312423503, 1436.8281919714111,
            1174.6110007708967, 1024.7571168934032
        ]
        deviation = 0.15  # 15%
        path = "../testbeds/{}/Parameters.properties".format(testbedName)

        # Various initial decisions for each scenario
        InitialRobotPos = [
            # 5 robots
            [
                [0.614911829488112, 0.8133550245441475],
                [0.005173486721660847, 0.8708144782022111],
                [0.7813029165430574, 0.7235740364415637],
                [0.6102258569310325, 0.6831150957320858],
                [0.307518832948647, 0.700133987499451]
            ],
            # 10 robots
            [
                [0.748782920935759, 0.523484207016743],
                [0.521725763522798, 0.662285767439358],
                [0.618314982927528, 0.850496608951951],
                [0.987035204212623, 0.683118335623724],
                [0.560976459754866, 0.480022865952500],
                [0.796815587856705, 0.712148754079348],
                [0.904113237958921, 0.006839213657844],
                [0.687208306090933, 0.641243548188644],
                [0.822574509070901, 0.141788922472766],
                [0.863995313984828, 0.247451873545336]
            ],
            # 15 robots
            [
                [0.14577420526318996, 0.576278714243178],
                [0.2287192796228119, 0.06335597195780196],
                [0.29834185863262586, 0.9628540563669717],
                [0.9931435119469243, 0.5494579142029646],
                [0.5462225494429768, 0.9426836186567821],
                [0.27633928302537847, 0.11416809281198337],
                [0.442823130636833, 0.12707121966850377],
                [0.023518096367830754, 0.3668238377023627],
                [0.2736123374545356, 0.42159878036987675],
                [0.13987608359458115, 0.7194024473396126],
                [0.9123916888742124, 0.6759966235761181],
                [0.5827780579029868, 0.7440031931387296],
                [0.6982890039215549, 0.5591813033016326],
                [0.5247276767048001, 0.17628825739163978],
                [0.7932215497908259, 0.8860535121525006]
            ],
            # 20 robots
            [
                [0.4302940553071771, 0.9336551468434044],
                [0.27904083669451984, 0.14355607681714966],
                [0.033174826178091976, 0.052435675144100524],
                [0.689857943305758, 0.9094755524109361],
                [0.016987800455446123, 0.502147576093662],
                [0.6729042340902326, 0.523797930027083],
                [0.5270870474503296, 0.6764456476961492],
                [0.1708931275935235, 0.8223618892029051],
                [0.7729124297670855, 0.6385338387254741],
                [0.9653252376509153, 0.4834610954269666],
                [0.9404450108644948, 0.5021734528331218],
                [0.25078272365907783, 0.2098095332401122],
                [0.31744707644427084, 0.8840957922768341],
                [0.5021017372382002, 0.4307793042384067],
                [0.9442617745352636, 0.9722833733626007],
                [0.6312799166890012, 0.14798769712391624],
                [0.2914232568431935, 0.044622775784423196],
                [0.3850766905967481, 0.22797575087108346],
                [0.43505529095951967, 0.593721720238489],
                [0.16961670019569774, 0.657633143182973]
            ]
        ]

        # Instantiate the correct testbed class
        testbed_class = self.testbeds_map[testbedName]

        for i, nrob in enumerate(NoRobots):
            props = self.rl.get_properties_ap(path)
            props["randomID"] = "true"
            props["displayTime&CF"] = "false"
            props["noRobots"] = str(nrob)
            Tmax = int(props["noIter"])

            # Create a fresh instance of the testbed for each run
            testbed_instance = testbed_class()

            # Pass the testbed instance to OptimizationLoop,
            # also supply initialDecisions.
            opt_loop = OptimizationLoop(
                testbed_instance,
                propertiesFILE=props,
                initialDecisions=InitialRobotPos[i]
            )
            curCF = opt_loop.getJJ()[Tmax - 1]
            print("[Number of robots: {}] Average CF value: {} | Recorded CF value: {}".format(
                nrob, DesiredCF[i], curCF))
            self.assertTrue(
                DesiredCF[i] * (1 - deviation) <= curCF <= DesiredCF[i] * (1 + deviation)
            )

if __name__ == '__main__':
    unittest.main()
