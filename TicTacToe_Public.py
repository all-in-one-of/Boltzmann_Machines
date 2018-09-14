#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 12:23:58 2018

@author: oster
"""
from tkinter import *
import TicTacToe as ttt


#run a public game
def publicGame():

    class App:

        def __init__(self, master):

            frame = Frame(master)
            frame.pack()

            self.button = Button(
                    frame, text="Quit", fg="red", command=frame.quit
                    )
            self.button.pack(side=LEFT)

            self.hi_there = Button(frame, text="Play Tic Tac Toe", command=self.openGame)
            self.hi_there.pack(side=LEFT)

        def openGame(self):
            print("hi there, everyone!")
            top = Toplevel()
            whichPlayer  = Frame(top)
            whichPlayer.pack()
            ChoosePlayer = Label(whichPlayer, text = "Do you wonna start first or second?")
            ChoosePlayer.pack()
            p1=Frame(top)
            p1.pack()
            player1 = Button(p1, text='I wonna be first', command = self.startGame)
            player1.pack(side=LEFT)
            p2 = Frame(top)
            p2.pack(side=RIGHT)
            player2 = Button(p1, text = 'I wonna play second')
            player2.pack(side=RIGHT)
            endGame =Frame(top)
            endGame.pack(side=BOTTOM)
            quitGame = Button(endGame, text='Quit', fg='red', command = top.quit)
            quitGame.pack()
            top.mainloop()
            top.destroy()
            
        def startGame(self):
            top = Toplevel()
            end = Frame(top)
            end.pack()
            quitGame = Button(end, text = "Quit", fg = 'red', command = top.quit)
            g1 = Frame(top).grid(row=0,column=1)
            mu = Label(g1,text='text1')
            mu.pack()
            g1.grid(row=0,column=0)
            g2=Frame(top)
            ma = Label(g2,text='ezt2')
            ma.pack()
          
            #e1= Entry(top)
            #e2= Entry(top)
            #e1.grid(row=0,column=0)
            #e2.grid(row=0, column=1)
            quitGame.grid(row=1,column=0)
            top.mainloop()
            top.destroy()
            
    root = Tk()

    App(root)

    root.mainloop()
    root.destroy()
publicGame()