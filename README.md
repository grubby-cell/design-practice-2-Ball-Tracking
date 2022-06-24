# Design Practice 2: NU G30 Engineering

## Project Description 
Ball tracking program for DP2 class. Used to trace ball across board using webcam feed or MP4 video file. Uses OpenCV module to identify and track ball via color selection and background subtraction methods, and logs ball data using custom data models created with Pydantic. Applies NumPy and SkLearn math to calculate physical movement properties.

## How to run program
**Install the necessary prerequisite modules**
- OpenCV
- NumPy
- Imutils
- MatplotLib
- PyTest
- Pydantic
- CSV
- Tabulate
- Datetime
- Scikit-Learn

**Clone the repository**
```
git clone https://github.com/jgfranco17/design-practice-2
```

**Execute main script**
Run the `main.py` file. Sample videos are already included in the repository. To change the object tracking, edit the parameters in the `balltrack.py` file.
