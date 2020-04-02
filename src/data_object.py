class DataObject:
    def __init__(self, image, mouse_position, screen_size, grid_size):
        self.image = image
        # x is for width, y is for height
        # (x, y) = mouse_position
        self.mouse_position = mouse_position
        # 0 is topleft, grid_size is topright, grid_size * grid_size is bottomright
        self.cell = self.determineCell(screen_size, grid_size)
        # # True if the click was really close to one of the corners
        # points = [(0, 0), (screen_size[0], 0),
        #           (0, screen_size[1]), screen_size]
        # closest_corner = (
        #     mouse_position[1] < screen_size[1]/2) * 2 + (mouse_position[0] < screen_size[0]/2)
        # distanceX = abs(mouse_position[0] - points[closest_corner][0])
        # distanceY = abs(mouse_position[1] - points[closest_corner][1])
        # self.is_close_to_corner = True if (
        #     distanceX <= screen_size[0]/4 and distanceY <= screen_size[1]/4) else False

    def determineCell(self, screen_size, grid_size):
        """
        Determines the cell on the grid of `grid_size`x`grid_size` that this image corresponds to.
        """
        dx = screen_size[0] / grid_size
        dy = screen_size[1] / grid_size
        return int(self.mouse_position[1] // dy * grid_size + self.mouse_position[0] // dx)

    def __str__(self):
        return f"Cursor: {self.mouse_position}, cell: {self.cell}"
