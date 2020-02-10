class DataObject:
    def __init__(self, image, mouse_position, screen_size):
        self.image = image
        # x is for width, y is for height
        # (x, y) = mouse_position
        self.mouse_position = mouse_position
        # 0 means up, 1 means down
        self.vertical = 0 if mouse_position[1] < screen_size[1]/2 else 1
        # 0 means left, 1 means down
        self.horizontal = 0 if mouse_position[0] < screen_size[0]/2 else 1
        # 0 is topleft, 1 is topright, 2 is bottomleft,  3 is bottomright
        self.square = self.vertical * 2 + self.horizontal
        # True if the click was really close to one of the corners
        points = [(0, 0), (screen_size[0], 0),
                  (0, screen_size[1]), screen_size]
        distanceX = abs(mouse_position[0] - points[self.square][0])
        distanceY = abs(mouse_position[1] - points[self.square][1])
        self.is_close_to_corner = True if (
            distanceX <= screen_size[0]/4 and distanceY <= screen_size[1]/4) else False

    def __str__(self):
        return f"Cursor: {self.mouse_position}, square: {self.square}"
