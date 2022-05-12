import numpy
import time
# import qlearn
import numpy as np
import random
import tensorflow as tf
from tkinter import *
import pylab


def nn_map_ploting(title, width, height, cell_size, obstacles, start, destination, nn,  map_dims, filename):

  window = Tk()
  window.title(title)
  canvas = Canvas(window, width=width, height=height)
  canvas.grid(row=0, column=0)

  # draw the grid
  j = 20
  for i in range(map_dims+1):
    canvas.create_line(20,j, cell_size*map_dims + 20,j, width = 1, fill = "black")
    j += cell_size
  j = 20
  for i in range(map_dims+1):
    canvas.create_line(j, 20, j, cell_size*map_dims+20, width = 1, fill = "black")
    j += cell_size
  
  # create a destination square on the map
  if len(destination) != 0:
    des_x = destination[0]*cell_size+ int(cell_size/2) + 20
    des_y = destination[1]*cell_size+ int(cell_size/2) + 20
    canvas.create_rectangle(des_x-int(cell_size/2), des_y-int(cell_size/2), des_x+int(cell_size/2), des_y+int(cell_size/2), fill = "red")

  # create the start point on the map:
  if len(start) != 0:
    start_x= start[0]*cell_size + int(cell_size/2) + 20
    start_y= start[1]*cell_size + int(cell_size/2) + 20
    canvas.create_rectangle(start_x-8, start_y-8, start_x+8, start_y+8, fill = "yellow")

  # draw the obstacles
  for i in range(len(obstacles)):
    dx1 = obstacles[i][0]*cell_size + 20
    dx2 = (obstacles[i][0]+1)*cell_size + 20
    dy1 = obstacles[i][1]*cell_size + 20
    dy2 = (obstacles[i][1]+1)*cell_size + 20
    
    rec1 = canvas.create_rectangle(dx1,dy1,dx2,dy2, fill = "black")

  if title =="ppo":
    for m in range(map_dims):
      for n in range(map_dims):
        if (m == destination[0]) and (n == destination[1]):
          pass
        elif (m,n) in obstacles:
          pass
        # elif (m == start[0]) and (n == start[1]):
        #   pass
        else:
          in_vector = np.zeros([1, 6*map_dims])
          in_vector[0,m] = 1.0
          in_vector[0,map_dims+n] = 1.0
          in_vector[0,2*map_dims+start[0]] = 1.0
          in_vector[0,3*map_dims+start[1]] = 1.0
          in_vector[0,4*map_dims+destination[0]] = 1.0
          in_vector[0,5*map_dims+destination[1]] = 1.0
          t = nn.predict(in_vector)
          direction = np.argmax(t[0])
          value = np.max(t[0])
          pos_x = 20 + m*cell_size
          pos_y = 20 + n*cell_size
          draw_arrow(canvas, direction, value, pos_x, pos_y, cell_size)
          canvas.pack()
          canvas.update() 
  elif title =="ppoper":
    for m in range(map_dims):
      for n in range(map_dims):
        if (m == destination[0]) and (n == destination[1]):
          pass
        elif (m,n) in obstacles:
          pass
        else:
          # in_vector = np.array([m,n], dtype=float)
          state1 = np.array((m,n))
          start1 = np.array((start[0],start[1]))
          target1 = np.array((destination[0],destination[1]))

          in_vector = decInputToBinInput(state_size=24, state=state1, start=start1, \
                                          target=target1, input_channels=2, n_bits=6)
          
          t = nn.predict(in_vector)
          t = np.exp(t[0])
          p = np.zeros(4)
          for i in range(len(p)):
            p[i] = t[i]/np.sum(t)
          # print(p)
          direction = np.argmax(p)
          value = np.max(p)
          pos_x = 20 + m*cell_size
          pos_y = 20 + n*cell_size
          draw_arrow(canvas, direction, value, pos_x, pos_y, cell_size)
          canvas.pack()
          canvas.update()

  canvas.postscript(file=filename, colormode='color')
  window.mainloop()


# convert a decimal position to a binary vector 
def decToBin(num, num_bit):
  vector = np.zeros(num_bit)
  i = num_bit -1
  while num > 1:
    vector[i] = num % 2
    num = num // 2
    i -= 1
  vector[i] = num
  return vector

# convert the decimal input vector into a binary input vector
def decInputToBinInput(state_size, state, start, target, input_channels, n_bits):
  state_convert = np.zeros([1, state_size])
  for i in range(input_channels):
    # for the current position
    if i == 0:
      binVec_x = decToBin(int(state[0]), n_bits)
      binVec_y = decToBin(int(state[1]), n_bits)    

    # for the target
    if i == 1:    
      binVec_x = decToBin(int(target[0]), n_bits)
      binVec_y = decToBin(int(target[1]), n_bits)
    # for the starting position
    if i == 2:
      binVec_x = decToBin(int(start[0]), n_bits)
      binVec_y = decToBin(int(start[1]), n_bits)          
    # save all into a vector
    state_convert[0,2*i*n_bits:(2*i+1)*n_bits] = binVec_x
    state_convert[0,(2*i+1)*n_bits:(2*i+2)*n_bits] = binVec_y
  
  return state_convert



def draw_arrow(canvas, direction, length, pos_x, pos_y, size):
    center_x = pos_x + int(0.5*size)
    center_y = pos_y + int(0.5*size)
    if direction == 2:  # up
        if length >= 0.7:
            canvas.create_line(center_x, center_y - int(size*0.31),center_x, center_y + int(size*0.31), fill='blue', arrow='first', width=0.5)
        else:
            canvas.create_line(center_x, center_y - int(size*0.15),center_x, center_y + int(size*0.15), fill='blue', arrow='first', width=0.5)

    if direction == 3: # down
        if length >= 0.7:
            canvas.create_line(center_x, center_y - int(size*0.31),center_x, center_y + int(size*0.33), fill='blue', arrow='last', width=0.5)
        else:
            canvas.create_line(center_x, center_y - int(size*0.15),center_x, center_y + int(size*0.15), fill='blue', arrow='last', width=0.5)

    if direction == 0: #'left':
        if length == 0.7:
            canvas.create_line(center_x - int(size*0.31), center_y ,center_x + int(size*0.33), center_y, fill='blue', arrow='first', width=0.5)
        else:
            canvas.create_line(center_x - int(size*0.15), center_y,center_x + int(size*0.15), center_y, fill='blue', arrow='first', width=0.5)

    if direction == 1: # 'right':
        if length == 0.7:
            canvas.create_line(center_x - int(size*0.31), center_y ,center_x + int(size*0.33), center_y, fill='blue', arrow='last', width=0.5)
        else:
            canvas.create_line(center_x - int(size*0.15), center_y,center_x + int(size*0.15), center_y, fill='blue', arrow='last', width=0.5)


class PlotModel():
  def __init__(self, path, figHeight=9, figWidth =18):
    self.scores = []
    self.episodes = []
    self.average = []
    self.path = path
    self.figHeight=figHeight
    self.figWidth=figWidth
    

  def plot_model(self, scores, n_episode):
    pylab.figure(figsize=(self.figWidth, self.figHeight))
    episodes_list =np.arange(n_episode, dtype=np.int32)
    average = []
    for i in range(n_episode):
      if i <=100:
        average.append(sum(scores[0:i+1])/len(scores[0:i+1]))
      else:
        average.append(sum(scores[i-100:i+1])/len(scores[i-100:i+1]))
    # for i in range(n_episode):
    # if i % 10 == 0:# much faster than episode % 100
    pylab.plot(episodes_list, scores, 'b')
    pylab.plot(episodes_list, average, 'r')
    pylab.ylabel('Score', fontsize=18)
    pylab.xlabel('Steps', fontsize=18)
    try:
      pylab.savefig(self.path+".png")
    except OSError:
      pass
    pylab.show()