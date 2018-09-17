#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 12:23:58 2018

@author: oster
"""
from tkinter import *
import TicTacToe as ttt
import numpy as np


#run a public game
def publicGame():

    class App:

        def __init__(self, master):
            
            ttt.training()

            frame = Frame(master)
            frame.pack()

            self.button = Button(
                    frame, text="Quit", fg="red", command=frame.quit
                    )
            self.button.pack(side=LEFT)

            self.hi_there = Button(frame, text="Play Tic Tac Toe", command=self.openGame)
            self.hi_there.pack(side=LEFT)

        def openGame(self):

            top = Toplevel()
            whichPlayer  = Frame(top)
            whichPlayer.pack()
            ChoosePlayer = Label(whichPlayer, text = "Do you wonna start first (1) or second (-1)?")
            ChoosePlayer.pack()
            p1=Frame(top)
            p1.pack()
            player1 = Button(p1, text='I wonna be first', command = self.startGame1)
            player1.pack(side=LEFT)
            p2 = Frame(top)
            p2.pack(side=RIGHT)
            player2 = Button(p1, text = 'I wonna play second', command = self.startGame2)
            player2.pack(side=RIGHT)
            endGame =Frame(top)
            endGame.pack(side=BOTTOM)
            quitGame = Button(endGame, text='Quit', fg='red', command = top.quit)
            quitGame.pack()
            top.mainloop()
            top.destroy()
            
        def startGame1(self):
            self.startGame(1)
        def startGame2(self):
            self.startGame(-1)
        def startGame(self,ply):
            top = Toplevel()
            
            self.state= np.matrix([[0,0,0],[0,0,0],[0,0,0]])

            self.quitGame = Button(top, text = "Quit", fg = 'red', command = top.quit)
            self.player = ply
            
            self.m00 = Button(top)
            self.m00.grid(row=0,column=0)
            self.m00['command'] = self.makeMove1

            self.m01 = Button(top)
            self.m01.grid(row=0,column=1)
            self.m01['command'] = self.makeMove2

            self.m02 = Button(top)
            self.m02.grid(row=0,column=2)
            self.m02['command'] = self.makeMove3

            self.m10 = Button(top)
            self.m10.grid(row=1,column=0)
            self.m10['command'] = self.makeMove4

            self.m11 = Button(top)
            self.m11.grid(row=1,column=1)
            self.m11['command'] = self.makeMove5

            self.m12 = Button(top)
            self.m12.grid(row=1,column=2)
            self.m12['command'] = self.makeMove6

            self.m20 = Button(top)
            self.m20.grid(row=2,column=0)
            self.m20['command'] = self.makeMove7

            self.m21 = Button(top)
            self.m21.grid(row=2,column=1)
            self.m21['command'] = self.makeMove8

            self.m22 = Button(top)
            self.m22.grid(row=2,column=2)
            self.m22['command'] = self.makeMove9
            
            self.buttons = [self.m00,self.m01,self.m02,self.m10,self.m11,self.m12,self.m20,self.m21,self.m22]
            self.buttons2 = ['self.m00','self.m01','self.m02','self.m10','self.m11','self.m12','self.m20','self.m21','self.m22']
            self.quitGame.grid(row=3,column=1)
            if ply==-1:
                self.state=ttt.executeMove(self.state,-1*ply,ttt.value_func)
                for i in range(3):
                    for j in range(3):
                        if self.state[i,j]!=0:
                            m='self.m'+str(i)+str(j)
                for i in range(9):
                    if self.buttons2[i]==m:
                        self.buttons[i].configure(text=str(-1*ply))
                
                        
            top.mainloop()
            
            top.destroy()
            
        def makeMove1(self):
            a=self.m00
            self.makeMove(a,[0,0])
        def makeMove2(self):
            a=self.m01
            self.makeMove(a,[0,1])
        def makeMove3(self):
            a=self.m02
            self.makeMove(a,[0,2])
        def makeMove4(self):
            a=self.m10
            self.makeMove(a,[1,0])
        def makeMove5(self):
            a=self.m11
            self.makeMove(a,[1,1])
        def makeMove6(self):
            a=self.m12
            self.makeMove(a,[1,2])
        def makeMove7(self):
            a=self.m20
            self.makeMove(a,[2,0])
        def makeMove8(self):
            a=self.m21
            self.makeMove(a,[2,1])
        def makeMove9(self):
            a=self.m22
            self.makeMove(a,[2,2])


        def makeMove(self,a,position):
            if a['text']=="":
                a['text']=str(self.player)
                
                self.state = ttt.move(self.state,[position[0],position[1], self.player])[0]
                if(ttt.winX(self.state)!=0):
                    top=Toplevel()
                    win=Label(top,text="Player "+str(self.player)+' won')
                    win.pack()
                    close = Button(top,text = "ok", command = top.quit)
                    close.pack()
                    top.mainloop()
                    self.quitGame.invoke()
                    top.destroy()
                    
                    return 0
                elif(np.count_nonzero(self.state)==9):
                    top=Toplevel()
                    win=Label(top,text="Both players won")
                    win.pack()
                    close = Button(top,text = "ok", command = top.quit)
                    close.pack()
                    top.mainloop()
                    self.quitGame.invoke()
                    top.destroy()
                    
                    return 0                    
                    
                dummy=self.state
                self.state = ttt.learn(self.state, ttt.value_func, -1*self.player)
              
                for i in range(3):
                    for j in range(3):
                        if (self.state-dummy)[i,j]!=0:
                            m='self.m'+str(i)+str(j)
                for i in range(9):
                    if self.buttons2[i]==m:
                        self.buttons[i].configure(text=str(-1*self.player))
                if(ttt.winX(self.state)!=0):
                    top=Toplevel()
                    win=Label(top,text="Player "+str(-1*self.player)+' won')
                    win.pack()
                    close = Button(top,text = "ok", command = top.quit)
                    close.pack()
                    top.mainloop()
                    top.destroy()
                    return 0  
                elif(np.count_nonzero(self.state)==9):
                    top=Toplevel()
                    win=Label(top,text="Both players won")
                    win.pack()
                    close = Button(top,text = "ok", command = top.quit)
                    close.pack()
                    top.mainloop()
                    self.quitGame.invoke()
                    top.destroy()
                    
                    return 0              
            else:
                top =Toplevel()
                error = Label(top, text = "Unfortunately, the field is already occupied")
                error.pack()
                quitError = Button(top, text = "Ok", command=top.quit)
                quitError.pack()
                top.mainloop()
                top.destroy()

    root = Tk()

    App(root)

    root.mainloop()
    root.destroy()
publicGame()