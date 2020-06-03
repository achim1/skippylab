"""
Plugins for different sensor types to be used with a Raspberry PI
Each plugin should provide a function which can pull data from 
a sensor attached to the Raspberry on the GPIO pins. The function
shall have the following form
func(*args,**kwargs) -> str
"""
