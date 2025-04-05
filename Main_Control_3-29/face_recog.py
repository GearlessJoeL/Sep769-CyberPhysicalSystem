from huskylib import HuskyLensLibrary

class FaceRecognition:
    def __init__(self):
        self.name = ""
    def recognize(self):
        hl = HuskyLensLibrary("I2C","", address=0x32)
        learned_blocks = hl.learnedBlocks()
        if learned_blocks == []:
            blocks = hl.blocks()
            if blocks == []:
                self.name = ""
            else:
                self.name = "Unknown"
        else:
            self.name = parse_recognition_result(learned_blocks)
    def get_name(self):
        return self.name
    def clear_name(self):
        self.name = ""


KNOWN_FACES = {
    1: "person1",
    2: "person2",
    3: "person3",
    4: "person4"
}

def parse_recognition_result(obj):
    """
    Parse the recognition result and assign names based on learned IDs
    
    Args:
        obj (list or object): HuskyLens recognition result
        
    Returns:
        dict: Parsed results with names and details
    """
    results = []
    
    if obj == []:
        return ""
    # Handle both single object and list of objects
    if not isinstance(obj, list):
        obj = [obj]
        
    for item in obj:
        # Convert object to dictionary for easier manipulation
        item_dict = item.__dict__
        
        # Try to get name from known faces/objects
        name = KNOWN_FACES.get(item_dict.get('ID', None), f"Unknown_{item_dict.get('ID', 'Object')}")
        
        return name