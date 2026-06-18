from enum import Enum


class GoniometerMode(Enum):
    MODE_362 = 362  # SIG x2 @296Hz, ACC 2g, GYRO 250dps
    MODE_363 = 363  # SIG x2 @296Hz, ACC 4g, GYRO 250dps
    MODE_364 = 364  # SIG x2 @296Hz, ACC 8g, GYRO 250dps
    MODE_365 = 365  # SIG x2 @296Hz, ACC 16g, GYRO 250dps
    MODE_366 = 366  # SIG x2 @296Hz, ACC 2g, GYRO 500dps
    MODE_367 = 367  # SIG x2 @296Hz, ACC 4g, GYRO 500dps
    MODE_368 = 368  # SIG x2 @296Hz, ACC 8g, GYRO 500dps
    MODE_369 = 369  # SIG x2 @296Hz, ACC 16g, GYRO 500dps
    MODE_370 = 370  # SIG x2 @296Hz, ACC 2g, GYRO 1000dps
    MODE_371 = 371  # SIG x2 @296Hz, ACC 4g, GYRO 1000dps
    MODE_372 = 372  # SIG x2 @296Hz, ACC 8g, GYRO 1000dps
    MODE_373 = 373  # SIG x2 @296Hz, ACC 16g, GYRO 1000dps
    MODE_374 = 374  # SIG x2 @296Hz, ACC 2g, GYRO 2000dps
    MODE_375 = 375  # SIG x2 @296Hz, ACC 4g, GYRO 2000dps
    MODE_376 = 376  # SIG x2 @296Hz, ACC 8g, GYRO 2000dps
    MODE_377 = 377  # SIG x2 @296Hz, ACC 16g, GYRO 2000dps
    MODE_378 = 378  # SIG x2 @370Hz, OR 32-bit @74Hz
    MODE_026 = 26  # 1 HF Chan @1926Hz, 1 LF Chan @148Hz
    MODE_244 = 244  # SIG x2 @519Hz

    def description(self):
        descriptions = {
            GoniometerMode.MODE_362: "SIG x2 @296Hz, ACC 2g, GYRO 250dps",
            GoniometerMode.MODE_363: "SIG x2 @296Hz, ACC 4g, GYRO 250dps",
            GoniometerMode.MODE_364: "SIG x2 @296Hz, ACC 8g, GYRO 250dps",
            GoniometerMode.MODE_365: "SIG x2 @296Hz, ACC 16g, GYRO 250dps",
            GoniometerMode.MODE_366: "SIG x2 @296Hz, ACC 2g, GYRO 500dps",
            GoniometerMode.MODE_367: "SIG x2 @296Hz, ACC 4g, GYRO 500dps",
            GoniometerMode.MODE_368: "SIG x2 @296Hz, ACC 8g, GYRO 500dps",
            GoniometerMode.MODE_369: "SIG x2 @296Hz, ACC 16g, GYRO 500dps",
            GoniometerMode.MODE_370: "SIG x2 @296Hz, ACC 2g, GYRO 1000dps",
            GoniometerMode.MODE_371: "SIG x2 @296Hz, ACC 4g, GYRO 1000dps",
            GoniometerMode.MODE_372: "SIG x2 @296Hz, ACC 8g, GYRO 1000dps",
            GoniometerMode.MODE_373: "SIG x2 @296Hz, ACC 16g, GYRO 1000dps",
            GoniometerMode.MODE_374: "SIG x2 @296Hz, ACC 2g, GYRO 2000dps",
            GoniometerMode.MODE_375: "SIG x2 @296Hz, ACC 4g, GYRO 2000dps",
            GoniometerMode.MODE_376: "SIG x2 @296Hz, ACC 8g, GYRO 2000dps",
            GoniometerMode.MODE_377: "SIG x2 @296Hz, ACC 16g, GYRO 2000dps",
            GoniometerMode.MODE_378: "SIG x2 @370Hz, OR 32-bit @74Hz",
            GoniometerMode.MODE_026: "1 HF Chan @1926Hz, 1 LF Chan @148Hz",
            GoniometerMode.MODE_244: "SIG x2 @519Hz",
        }
        return descriptions.get(self, "Unknown Mode")
