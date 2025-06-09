import pygame as pg
import numpy as np

def main():
	pg.mixer.init()  # Inicializar el sistema de sonido
	pg.init()
	pg.mixer.music.load('musica.mp3')  # Carga el archivo MP3 (debe estar en la misma carpeta)
	pg.mixer.music.play(-1)  # Reproduce en bucle

	screen = pg.display.set_mode((800,600))
	running=True
	clock = pg.time.Clock()

	hres=120 #horizontal
	halfvres = 100 #vertical/2

	mod = hres/60 #factor de escalado
	posx, posy, rot = 0, 0, 0
	frame = np.random.uniform(0,1,(hres,halfvres*2,3))
	sky = pg.image.load('skybox1.jpg')  #360,100 pixels
	sky = pg.surfarray.array3d(pg.transform.scale(sky, (360, halfvres*2)))
	floor = pg.surfarray.array3d(pg.image.load('floor3.jpg'))

	while running:
		for event in pg.event.get():
			if event.type == pg.QUIT:
				running = False

		for i in range(hres):
			rot_i = rot + np.deg2rad(i/mod-30)
			sin,cos,cos2 = np.sin(rot_i), np.cos(rot_i), np.cos(np.deg2rad(i/mod-30))
			frame[i][:] = sky[int(np.rad2deg(rot_i)%359)][:]/255
			for j in range(halfvres):
				n = (halfvres/(halfvres-j))/cos2
				x,y = posx+cos*n, posy+sin*n 
				xx,yy = int(x*2%1*99), int(y*2%1*99)
				shade = 0.2 + 0.8 * (1-j/halfvres)
				frame[i][halfvres*2-j-1] = shade*floor[xx][yy]/255

		surf=pg.surfarray.make_surface(frame*255)
		surf=pg.transform.scale(surf, (800,600))

		screen.blit(surf,(0,0))
		pg.display.update()
		posx, posy, rot = movement(posx, posy,rot, pg.key.get_pressed())

def movement(posx,posy,rot,keys):
	if keys[pg.K_LEFT] or keys[ord('a')]:
		rot = rot - 0.1
	if keys[pg.K_RIGHT] or keys[ord('d')]:
		rot = rot + 0.1
	if keys[pg.K_UP] or keys[ord('w')]:
		posx, posy=posx+np.cos(rot)*0.1, posy+np.sin(rot)*0.1		
	if keys[pg.K_DOWN] or keys[ord('s')]:
		posx, posy=posx-np.cos(rot)*0.1, posy-np.sin(rot)*0.1		
	return posx,posy,rot

if __name__ == '__main__':
	main()
	pg.quit()
