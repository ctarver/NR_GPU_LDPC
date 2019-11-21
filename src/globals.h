/**
    globals.h
    Stuff I want to get rid of but haven't quite yet.

    @author Chance Tarver
    @version 0.1 11/18/19
*/

#ifndef GLOBALS
#define GLOBALS

#define HALF_PRC 0
#define CHAR_PRC 1

#define BG1  // Choose BG1 or 2.

#ifdef BG1
#define BLK_ROW 46
#define BLK_COL 68
#define NON_EMPTY_ROW 19
#define NON_EMPTY_COL 30

#else
#define BLK_ROW 42
#define BLK_COL 52
#define NON_EMPTY_ROW 10
#define NON_EMPTY_COL 23
#endif

#define H_COMPACT1_ROW BLK_ROW
#define H_COMPACT1_COL NON_EMPTY_ROW
#define H_COMPACT2_ROW NON_EMPTY_COL
#define H_COMPACT2_COL BLK_COL

#endif
