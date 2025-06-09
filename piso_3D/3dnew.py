import pygame as pg
import numpy as np

def main():
    pg.init()
    screen = pg.display.set_mode((800, 600))
    running = True
    clock = pg.time.Clock()

    hres = 120  # horizontal resolution (columns)
    halfvres = 100  # half vertical resolution (rows)

    mod = hres / 60  # projection scale
    posx, posy, rot = 10, 10, 0  # initial player position and rotation
    frame = np.zeros((hres, halfvres * 2, 3), dtype=np.float32)

    # Load sky and scale
    sky = pg.image.load('atardecer.jpg')  # assumed 360x100
    sky = pg.surfarray.array3d(pg.transform.scale(sky, (360, halfvres * 2)))

    # Load floor image (full map)
    floor_image = pg.image.load('ruta_info.png')
    floor = pg.surfarray.array3d(floor_image)
    floor_width, floor_height = floor.shape[0], floor.shape[1]

    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        for i in range(hres):
            angle = np.deg2rad(i / mod - 30)
            rot_i = rot + angle
            sin, cos = np.sin(rot_i), np.cos(rot_i)
            cos2 = np.cos(angle)

            # sky column
            sky_x = int(np.rad2deg(rot_i) % 360)
            frame[i][:] = sky[sky_x][:] / 255

            for j in range(halfvres):
                n = (halfvres / (halfvres - j)) / cos2
                x = posx + cos * n
                y = posy + sin * n

                # Convert world coords to image coords
                xx = int(x * (floor_width / 50))  # world units to pixel scale
                yy = int(y * (floor_height / 50))

                # Clipping to avoid out-of-bounds
                if 0 <= xx < floor_width and 0 <= yy < floor_height:
                    shade = 0.2 + 0.8 * (1 - j / halfvres)
                    frame[i][halfvres * 2 - j - 1] = shade * floor[xx][yy] / 255
                else:
                    frame[i][halfvres * 2 - j - 1] = [0, 0, 0]  # black if out of bounds

        # Render and scale to screen
        surf = pg.surfarray.make_surface(np.clip(frame * 255, 0, 255).astype(np.uint8))
        surf = pg.transform.scale(surf, (800, 600))
        screen.blit(surf, (0, 0))
        pg.display.update()

        posx, posy, rot = movement(posx, posy, rot, pg.key.get_pressed())
        clock.tick(60)

def movement(posx, posy, rot, keys):
    if keys[pg.K_LEFT] or keys[ord('a')]:
        rot -= 0.08
    if keys[pg.K_RIGHT] or keys[ord('d')]:
        rot += 0.08
    if keys[pg.K_UP] or keys[ord('w')]:
        posx += np.cos(rot) * 0.5
        posy += np.sin(rot) * 0.5
    if keys[pg.K_DOWN] or keys[ord('s')]:
        posx -= np.cos(rot) * 0.5
        posy -= np.sin(rot) * 0.5
    return posx, posy, rot

if __name__ == '__main__':
    main()
    pg.quit()
