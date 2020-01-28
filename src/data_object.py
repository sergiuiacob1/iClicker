class DataObject:
    def __init__(self, image, mouse_position):
        self.image = image
        self.mouse_position = mouse_position
        # x is for width, y is for height
        # (x, y) = mouse_position
        # self.horizontal = 0 if self.mouse_position[0] < self.screenSize[0]/2 else 1
        # self.vertical = 0 if self.mouse_position[1] < self.screenSize[1]/2 else 1

    def __str__(self):
        return f'Image Size: {len(self.image)}, mouse position: {self.mouse_position}'
