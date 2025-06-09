import pygame
import sys

# Inicializar Pygame
pygame.init()

# Configuración de pantalla
ANCHO, ALTO = 1100, 600
pantalla = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("Tarjeta Animada con Sprite")
reloj = pygame.time.Clock()

sprite_sheet = pygame.image.load("dragon.png").convert_alpha()
#sprite_sheet.set_colorkey((255, 255, 255))  # Hace el blanco transparente
num_frames = sprite_sheet.get_width() // 169
sprites = [sprite_sheet.subsurface(pygame.Rect(i * 169, 0, 169, 269)) for i in range(num_frames)]

# Fuente y textos
fuente = pygame.font.SysFont("arial", 32, bold=True)
textos = [
    "Codigo 233",
    "Cristhian Andres ",
    "Escobar Herrera"
]
textos_render = [fuente.render(t, True, (255, 255, 255)) for t in textos]
textos_rects = [tr.get_rect(center=(ANCHO // 2, ALTO // 2 + i * 60)) for i, tr in enumerate(textos_render)]


x = 0  # inicio fuera de la pantalla
y = textos_rects[0].top - 260  # justo encima del primer texto
frame = 0
vel = 6

# Control de tiempo para mostrar los textos progresivamente
tiempo_inicio = pygame.time.get_ticks()
mostrar_segundo = False
mostrar_tercero = False

# Bucle principal
while True:
    pantalla.fill((33, 33, 33))

    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Mostrar textos según el tiempo
    ahora = pygame.time.get_ticks()
    pantalla.blit(textos_render[0], textos_rects[0])  # Siempre mostrar el nombre

    if ahora - tiempo_inicio > 2000:
        pantalla.blit(textos_render[1], textos_rects[1])
        mostrar_segundo = True

    if ahora - tiempo_inicio > 4000:
        pantalla.blit(textos_render[2], textos_rects[2])
        mostrar_tercero = True

    # Dibujar perrito y animar
    pantalla.blit(sprites[frame // 5], (x, y))
    frame = (frame + 1) % (num_frames * 5)

    x += vel
    if x > ANCHO:
        x = -169  # reinicia el camino

    pygame.display.update()
    reloj.tick(20)