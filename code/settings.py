class Settings(object):
    CHUNK = 512
    RATE = 44100
    
    FILE_DURATION = 10      # seconds
    MIN_FILE_DURATION = 2   # seconds

    DETECTION_WINDOW = 0.1  # seconds
    DETECTION_BUFFER = 0.5  # seconds
    DETECTION_AMP_THRESHOLD = 100
    DETECTION_CROSSINGS_PER_CHUNK = 40
