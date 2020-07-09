Usage info for blender 2.83.1 

1) Start blender *via terminal* (to see logs)

2) Load mesh: File (located in Header Bar) -> Import -> Stanford (.ply)

3) Open scripts:
   3a) Click on "Scripting" (located in Header Bar)
   3b) Text (located in Header Bar of opened Text Editor window in the middle) -> Open

-> To get object id for selection:
1) Enter Edit Mode (modes are located top left)
2) Set selection type as "Face Select" (located on the right of the dropdown menu used to select Edit Mode)
3) Select some faces
4) Open "obj_id_for_selection.py" script from the Text Editor window. 
5) Update ASCII_PATH variable with the ascii version of the ply file to be manipulated 
6) Run script: Output should be logged on terminal running blender, *not on its internal python console*. 

-> To select faces of an object with certain object id:
1) Open "select_by_obj_id.py" script from the Text Editor window. 
2) Update ASCII_PATH variable with the ascii version of the ply file to be manipulated 
3) Update SELECT_OBJ variable with a valid object ID
4) Run script: Selection should be reflected on GUI.