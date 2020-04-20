def animate(folder_with_frames, output_filename='animation.mp4', num_frames = 1, fps=1, forward_reverse=False):
    import os
    import imageio

    # imageio.plugins.ffmpeg.download() # <-- you may need to run this the first time you use this code
    with imageio.get_writer(os.path.normpath(output_filename), fps=fps) as writer:


        # for frame in sorted(os.listdir(folder_with_frames)):
        for i in range(num_frames):
            frame = str(i) + '.png'
            if frame.endswith('.png') or frame.endswith('.jpg') or frame.endswith('.jpeg'):
                image = imageio.imread(os.path.normpath(os.path.join(folder_with_frames, frame)))
                writer.append_data(image)


        if forward_reverse == True:

            # for frame in sorted(os.listdir(folder_with_frames), reverse=True):
            for i in range(num_frames-1,0,-1):
                frame = str(i) + '.png'
                if frame.endswith('.png') or frame.endswith('.jpg') or frame.endswith('.jpeg'):
                    image = imageio.imread(os.path.normpath(os.path.join(folder_with_frames, frame)))
                    writer.append_data(image)



if __name__ == '__main__':
    folder_with_frames = 'animation_frames/'
    animate(
            folder_with_frames,
            num_frames = 10,
        )
