import numpy as np
import cv2 as cv
import tifffile as tiff
import matplotlib.pyplot as plt
import os
import io
import PySimpleGUI as sg
from PIL import Image

# Image class from object based processing
class MyImage():
    def __init__(self, img, low=None, high=None):
        self.img = img
        self.isolate = None
        self.mask = None
        self.G = None
        self.S = None
        self.Mod = None
        self.Ph = None
        self.low = low
        self.high = high

    # phasor function calculate values
    def calculate_phasors(self, img=None):
        if img is None:
            img = self.img

        fft=np.fft.fft(img, axis=2)
    
        G=fft[:,:,1].real/fft[:,:,0].real
        G=np.nan_to_num(G, nan=0.0)
    
        S=fft[:,:,1].imag/fft[:,:,0].real
        S=np.nan_to_num(S, nan=0.0)
    
        Ph=np.arctan2(S[:,:], G[:,:])+np.pi
        Ph=np.nan_to_num(Ph, nan=0.0)
    
        Mod=np.sqrt(G**2+S**2)
        Mod=np.nan_to_num(Mod, nan=0.0)
    
        self.G = G
        self.S = S
        self.Ph = Ph
        self.Mod = Mod

    # plots each value separately
    def plot_phasors(self):
        plt.figure()
        plt.imshow(self.Mod)
        # plt.axis('off')
        plt.title('Mod')
        plt.colorbar()

        plt.figure()
        plt.imshow(self.Ph)
        # plt.axis('off')
        plt.title('Phase')
        plt.colorbar()

        plt.figure()
        plt.imshow(self.S)
        # plt.axis('off')
        plt.title('S')
        plt.colorbar()

        plt.figure()
        plt.imshow(self.G)
        # plt.axis('off')
        plt.title('G')
        plt.colorbar()

        plt.show()
    
    # define rescaling function
    def rescale_frame(self, scale=0.5):
        width = int(self.img.shape[1] * scale)
        height = int(self.img.shape[0] * scale)

        dimensions = (width, height)
        return cv.resize(self.img, dimensions, interpolation=cv.INTER_AREA)
    
    # separate rescaling function that fixes the maximum size to 700x700
    def rescale_fixed(self):
        width = int(self.img.shape[1])
        height = int(self.img.shape[0])

        larger_dimension = max(width, height)

        rescaled_width = int(width * (700 / larger_dimension))
        rescaled_height = int(height * (700 / larger_dimension))

        dimensions = (rescaled_width, rescaled_height)
        return cv.resize(self.img, dimensions, interpolation=cv.INTER_AREA), rescaled_width, rescaled_height, larger_dimension

    # Setters
    def set_mask(self):
        # Using inRange method, to create a mask
        self.mask = cv.inRange(self.img, self.low, self.high)

    def set_isolate(self):
        self.isolate = cv.bitwise_and(self.img, self.img, mask=self.mask)

    # Getters
    def get_mask(self):
        return self.mask
    
    def get_img(self):
        return self.img
    
    def get_isolate(self):
        return self.isolate
    
    def get_g(self):
        return self.G
    
    def get_s(self):
        return self.S
    
    def get_mod(self):
        return self.Mod
    
    def get_ph(self):
        return self.Ph
    
    def get_low(self):
        return self.low
    
    def get_high(self):
        return self.high

# calculate top left and bottom right corners given any two corners
def calculate_corners(corner1, corner2):
    top_left = (min(corner1[0],corner2[0]), max(corner1[1],corner2[1]))
    bot_right = (max(corner1[0],corner2[0]), min(corner1[1],corner2[1]))
    return top_left, bot_right

# GUI wrapper for viewing phasors

file_types = [("JPEG (*.jpg)", "*.jpg"),
              ("TIFF (*.tif)", "*.tif"),
              ("PNG (*.png)", "*.png"),
              ("All files (*.*)", "*.*")]

def main():
    lowSliders = [
        [   
            # Hue Slider
            sg.Checkbox("Hue", "Radio", size=(10, 1), key="-HSV-"),
            sg.Slider(
                (0,179),
                90,
                1,
                orientation='horizontal',
                size=(30,15),
                key="-LO HUE SLIDER-"      
            )
        ],
        [
            # Low Sat slider
            sg.Checkbox("Sat", "Radio2", size=(10, 1), key="-HSV-"),
            sg.Slider(
                (0,255),
                90,
                1,
                orientation='horizontal',
                size=(30,15),
                key="-LO SAT SLIDER-" 
            )
        ],
        [
            # Low Value Slider
            sg.Checkbox("Val", "Radio3", size=(10, 1), key="-HSV-"),
            sg.Slider(
                (0,255),
                90,
                1,
                orientation='horizontal',
                size=(30,15),
                key="-LO VAL SLIDER-" 
            )
        ]
    ]

    highSliders = [
        [
            # High Hue slider
            sg.Checkbox("Hue", "Radio4", size=(10, 1), key="-HSV-"),
            sg.Slider(
                (0,179),
                179,
                1,
                orientation='horizontal',
                size=(30,15),
                key="-HI HUE SLIDER-"      
            )
        ],
        [
            # High Sat slider
            sg.Checkbox("Sat", "Radio5", size=(10, 1), key="-HSV-"),
            sg.Slider(
                (0,255),
                255,
                1,
                orientation='horizontal',
                size=(30,15),
                key="-HI SAT SLIDER-" 
            )
        ],
        [
            # High Value Slider
            sg.Checkbox("Val", "Radio6", size=(10, 1), key="-HSV-"),
            sg.Slider(
                (0,255),
                255,
                1,
                orientation='horizontal',
                size=(30,15),
                key="-HI VAL SLIDER-" 
            )
        ]
    ]

    blurSlider = [
        [
            # Blur Slider
            sg.Checkbox("Blur", size=(10, 1), key="-BLUR-"),
            sg.Slider(
                (3,13),
                3,
                2,
                orientation='horizontal',
                size=(20,15),
                key="-BLUR SLIDER-"      
            )
        ]
    ]

    draw_col = [
            [sg.Checkbox("Apply Mask", size=(10, 1), key="-MASK-")],
            [sg.Radio('Draw Rectangle', 'Mask Options', 1, key='-RECT-', enable_events=True)],
            [sg.Radio('Move Rectangle', 'Mask Options', 0, key='-MOVE-', enable_events=True)],
            [sg.Radio('Erase Rectangle', 'Mask Options', 0, key='-ERASE-', enable_events=True)],
        ]

    file_list_column = [
        [
            sg.Text("Image File"),
            sg.Input(size=(25, 1), enable_events=True, key="-FILE-"),
            sg.FileBrowse(file_types=file_types),
            sg.Button("Load Image"),
            sg.Button("Apply Settings"),
            sg.Button("Generate Phasor Plots"),
            sg.Button('Save Image', key='-SAVE-')
        ]
    ]
    graph_settings = sg.Graph(
                canvas_size=(0,0),
                graph_bottom_left=(0, 0),
                graph_top_right=(700, 700),
                key="-GRAPH-",
                enable_events=True,
                background_color='lightblue',
                drag_submits=True,
                motion_events=True,
            )

    layout = [
        [
            # Image viewer
            graph_settings,
            sg.Frame("Mask Options", draw_col)
        ],
        [sg.Text(key='-INFO-', size=(60, 1))
        ],
        [
            sg.Column(file_list_column)
        ],
        [
            sg.Column(blurSlider)
        ],
        [   sg.Frame("Lows", lowSliders),
            sg.VSeperator(),
            sg.Frame("Highs", highSliders)
        ]
    ]

    window = sg.Window("Phasor Viewer", layout)

    filename = ''
    

    graph = window["-GRAPH-"]       # type: sg.Graph
    dragging = False
    start_point = end_point = prior_rect = former_start = former_end = None

    while True:
        HSV_flag = False
        if filename != '':
            frame = cv.imread(filename=filename)

        event, values = window.read(timeout=20)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        if event == "Load Image":
            # if you load image twice in a row then it doesnt work - fixed with if statement
            # this one need to resize the thing also - fixed with change coordinates
            try:
                if 'back' in locals():
                    graph.delete_figure(back)
                
                filename = values["-FILE-"]
                frame = cv.imread(filename=filename)

                show_frame = MyImage(frame)
                show_frame, width, height, larger_dim = show_frame.rescale_fixed()
                graph_settings.set_size((width, height))
                graph_settings.change_coordinates((0,0), (width, height))
                imencode = cv.imencode(".png", show_frame)[1]
                imgbytes = np.array(imencode).tobytes()
                back = graph.draw_image(data=imgbytes, location=(0,700))
            except (UnboundLocalError, AttributeError):
                pass
        
        if values["-BLUR-"]:
            try:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                HSV_flag = True

                frame = cv.medianBlur(frame, int(values["-BLUR SLIDER-"]))
            except (UnboundLocalError, cv.error):
                pass
        else:
            HSV_flag = False

        if values["-HSV-"]:
            try:
                if HSV_flag is False:
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                    HSV_flag = True

                image_frame = MyImage(frame, 
                                low=np.array([values["-LO HUE SLIDER-"], values["-LO SAT SLIDER-"], values["-LO VAL SLIDER-"]]),
                                high=np.array([values["-HI HUE SLIDER-"], values["-HI SAT SLIDER-"], values["-HI VAL SLIDER-"]]))
                image_frame.set_mask()
                image_frame.set_isolate()
                frame = image_frame.get_isolate()
                
            except (UnboundLocalError, cv.error):
                pass
                
            except (UnboundLocalError, AttributeError, TypeError) as error:
                print(error)

        if values["-MASK-"]:
            try:
                if 'former_start' in locals():
                    top_left, bot_right = calculate_corners(former_start, former_end)

                    # since the rectangle is drawn on the small one we have to recalculate the big one to put the mask
                    # for some reason the x coordinate is always correct but the height/width are wrong
                    # first convert to opencv coordinates from sg coordinates and then resize
                    # for opencv, (0,0) is top left and (x,x) is bottom right
                    # for pysimple gui, (0,0) is bottom left and (x,x) is top right
                    # to convert y would be 700 - y
                    # don't need to convert x?
                    # - fixed
                    top_left = tuple((top_left[0], (700 - top_left[1])))
                    bot_right = tuple((bot_right[0], (700 - bot_right[1])))
                    top_left = tuple((int(top_left[0] * (larger_dim / 700)), int(top_left[1]  * (larger_dim / 700))))
                    bot_right = tuple((int(bot_right[0] * (larger_dim / 700)), int(bot_right[1] * (larger_dim / 700))))
                    blank = np.zeros(frame.shape[:2], dtype='uint8')
                    rectangle = cv.rectangle(blank, top_left, bot_right, 255, -1)
                    frame = cv.bitwise_and(frame, frame, mask=rectangle)
            except (UnboundLocalError, TypeError) as error:
                pass

        if event == "Generate Phasor Plots":
            plt.close('all')
            try:
                phasor_frame = MyImage(cv.cvtColor(frame, cv.COLOR_HSV2BGR))
                phasor_frame.calculate_phasors(phasor_frame.isolate)
                phasor_frame.plot_phasors()
            except Exception:
                pass

        if event == "Apply Settings":
            try:
                graph.delete_figure(back)
                show_frame = MyImage(frame)
                show_frame, width, height, larger_dim = show_frame.rescale_fixed()
                show_frame = cv.cvtColor(show_frame, cv.COLOR_HSV2BGR)
                graph_settings.set_size((width, height))
                graph_settings.change_coordinates((0,0), (width, height))
                imencode = cv.imencode(".png", show_frame)[1]
                imgbytes = np.array(imencode).tobytes()
                back = graph.draw_image(data=imgbytes, location=(0,700))
                graph.send_figure_to_back(back)
            except (UnboundLocalError, AttributeError):
                pass
            # the graph size is getting resized but the top left and right arent gettign resized - fixed with change coordinates

        elif event == '-SAVE-':
            try:
                save_frame = cv.cvtColor(frame, cv.COLOR_HSV2BGR)
                # filename = sg.popup_get_file('Choose file (PNG, JPG, GIF) to save to', save_as=True)
                filename = sg.popup_get_file('Click Save As and choose file name (include extension)', save_as=True)
                dirs = filename.rsplit('/', 1)
                dir = dirs[0]
                name = dirs[1]
                os.chdir(dir) 
                cv.imwrite(name, save_frame) 
            except (ValueError, UnboundLocalError, AttributeError, IndexError):
                print("Unknown File Extension")

        if event in ('-MOVE-', '-MOVEALL-'):
            graph.set_cursor(cursor='fleur')          # not yet released method... coming soon!
        elif not event.startswith('-GRAPH-'):
            graph.set_cursor(cursor='left_ptr')       # not yet released method... coming soon!
        if event.endswith('+MOVE'):
            window["-INFO-"].update(value=f"mouse {values['-GRAPH-']}")

        if event == "-GRAPH-":  # if there's a "Graph" event, then it's a mouse
            x, y = values["-GRAPH-"]
            if not dragging:
                start_point = (x, y)
                dragging = True
                lastxy = x, y
            else:
                end_point = (x, y)
            if prior_rect:
                graph.delete_figure(prior_rect)
            delta_x, delta_y = x - lastxy[0], y - lastxy[1]
            lastxy = x,y
            if None not in (start_point, end_point):
                # move only moves one rectangle
                if values['-MOVE-']:
                    if 'old_rect' in locals():
                        graph.move_figure(old_rect, delta_x, delta_y)
                        graph.update()
                    else:
                        pass
                # only one rectangle should be possible to exist
                elif values['-RECT-']:
                    if 'old_rect' in locals():
                        graph.delete_figure(old_rect)
                    prior_rect = graph.draw_rectangle(start_point, end_point, line_color='red')
                # erase should only erase one rectange
                elif values['-ERASE-']:
                    if 'old_rect' in locals():
                        graph.delete_figure(old_rect)
                    else:
                        pass

            window["-INFO-"].update(value=f"mouse {values['-GRAPH-']}")
        elif event.endswith('+UP'):         # The drawing has ended because mouse up
            window["-INFO-"].update(value=f"grabbed rectangle from {start_point} to {end_point}")
            former_start, former_end = start_point, end_point
            start_point, end_point = None, None  # enable grabbing a new rect
            dragging = False
            if prior_rect is not None:
                old_rect = prior_rect
            prior_rect = None
        # elif event.endswith('+RIGHT+'):     # Right click
        #     window["-INFO-"].update(value=f"Right clicked location {values['-GRAPH-']}")
        # elif event.endswith('+MOTION+'):    # Right click
        #     window["-INFO-"].update(value=f"mouse freely moving {values['-GRAPH-']}")

    window.close()

if __name__ == '__main__':
    main()