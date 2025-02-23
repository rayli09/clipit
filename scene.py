from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.openai import OpenAIService
import numpy as np

# Helper function to fade out all mobjects in a scene
def fade_out(scene: Scene):
    if not scene.mobjects:
        return
    animations = []
    for mobject in scene.mobjects:
        animations.append(FadeOut(mobject))
    scene.play(*animations)

class ProblemIntroduction(VoiceoverScene):
    def construct(self):
        # Set the default font
        Text.set_default(font="Source Code Pro")
        self.set_speech_service(OpenAIService(voice="fable", model="tts-1-hd"))
        
        # Title and problem description
        title = Tex(r"Elevator Stop Problem", font_size=48)
        problem_text = Tex(
            r"A building has 10 floors above the basement. 12 people get in the elevator,",
            r"and each chooses a floor uniformly at random.",
            r"Find the expected number of stops.",
            font_size=36
        )
        VGroup(title, problem_text).arrange(DOWN, buff=0.5).to_edge(UP)
        
        # A simple diagram: 10 floors represented by horizontal lines
        floors = VGroup(*[
            Line(LEFT, RIGHT).scale(2).shift(DOWN * i)
            for i in range(5)
        ])
        floors.to_edge(LEFT, buff=1)
        floor_labels = VGroup(*[
            Text(f"Floor {i+1}", font_size=24).next_to(floor, LEFT)
            for i, floor in enumerate(floors)
        ])
        diagram = VGroup(floors, floor_labels).to_edge(DOWN, buff=1)
        
        self.play(Write(title), run_time=1)
        with self.voiceover(text="We have a building with 10 floors above the basement and 12 people entering the elevator at the basement. Each person randomly chooses a floor to get off."):
            self.play(Write(problem_text), run_time=3)
        self.play(Create(diagram), run_time=2)
        self.wait(1)

class IndicatorVisualization(VoiceoverScene):
    def construct(self):
        # Set the default font
        Text.set_default(font="Source Code Pro")
        self.set_speech_service(OpenAIService(voice="fable", model="tts-1-hd"))
        
        # Title for this part
        title = Tex(r"Indicator Variables", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Visual: Represent each floor with a box and its indicator variable X_i
        floor_boxes = VGroup(*[
            Square(side_length=0.8, color=BLUE).shift(RIGHT * i)
            for i in range(10)
        ]).arrange(RIGHT, buff=0.5).to_edge(DOWN, buff=1)
        
        labels = VGroup(*[
            MathTex(r"X_{" + f"{i+1}" + r"}").next_to(floor, DOWN, buff=0.2)
            for i, floor in enumerate(floor_boxes)
        ])
        
        self.play(Create(floor_boxes), Write(labels), run_time=2)
        with self.voiceover(text="We assign an indicator variable to each floor. For floor i, X_i equals 1 if at least one person gets off there, and 0 otherwise."):
            self.wait(3)
        
        # Highlight one floor to show the idea
        highlight = SurroundingRectangle(floor_boxes[3], color=YELLOW)
        self.play(Create(highlight))
        with self.voiceover(text="For example, if someone gets off on floor 4, then X_4 is 1. Otherwise, it remains 0."):
            self.wait(3)
        self.play(FadeOut(highlight))

class CalculationScene(VoiceoverScene):
    def construct(self):
        # Set the default font
        Text.set_default(font="Source Code Pro")
        self.set_speech_service(OpenAIService(voice="fable", model="tts-1-hd"))
        
        # Title for calculation
        title = Tex(r"Calculating the Expected Stops", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Step 1: Expressing the total stops S as the sum of indicators
        expr1 = MathTex(r"S = X_1 + X_2 + \cdots + X_{10}", font_size=42)
        expr1.next_to(title, DOWN, buff=0.8)
        self.play(Write(expr1), run_time=2)
        with self.voiceover(text="We express the total number of stops, S, as the sum of the indicator variables for each floor."):
            self.wait(3)
        
        # Step 2: Using linearity of expectation
        expr2 = MathTex(r"E[S] = E[X_1] + E[X_2] + \cdots + E[X_{10}]", font_size=42)
        expr2.next_to(expr1, DOWN, buff=0.8)
        self.play(Write(expr2), run_time=2)
        with self.voiceover(text="By the linearity of expectation, the expected number of stops is the sum of the expected values of each indicator."):
            self.wait(3)
        
        # Step 3: Calculate E[X_i]
        expr3 = MathTex(
            r"E[X_i] = P(X_i=1) = 1 - P(\text{no one gets off at floor } i)",
            font_size=36
        )
        expr3.next_to(expr2, DOWN, buff=0.8)
        self.play(Write(expr3), run_time=3)
        with self.voiceover(text="Since the expected value of an indicator is just the probability it is 1, we need the probability that at least one person gets off on a floor."):
            self.wait(3)
        
        # Step 4: Compute probability that no one gets off on a given floor
        expr4 = MathTex(r"P(X_i=0)=\left(\frac{9}{10}\right)^{12}", font_size=42)
        expr4.next_to(expr3, DOWN, buff=0.8)
        self.play(Write(expr4), run_time=2)
        with self.voiceover(text="Each person has a 9 in 10 chance of not choosing a given floor, so for 12 people, the probability that no one gets off is (9/10) to the power 12."):
            self.wait(4)
        
        # Step 5: Putting it all together
        expr5 = MathTex(
            r"E[S]=10\left(1-\left(\frac{9}{10}\right)^{12}\right)",
            font_size=42
        )
        expr5.next_to(expr4, DOWN, buff=0.8)
        self.play(Write(expr5), run_time=3)
        with self.voiceover(text="Thus, the expected number of stops is 10 times 1 minus (9/10) raised to the 12th power."):
            self.wait(3)
        
        # Step 6: Numerical Approximation
        approx_value = 10 * (1 - (9/10)**12)
        expr6 = MathTex(r"\approx " + f"{approx_value:.2f}", font_size=42)
        expr6.next_to(expr5, DOWN, buff=0.8)
        self.play(Write(expr6), run_time=2)
        with self.voiceover(text=f"Evaluating this expression, we find that the elevator is expected to stop at approximately {approx_value:.2f} floors."):
            self.wait(4)

class CombinedScene(VoiceoverScene):
    def construct(self):
        scenes:list[VoiceoverScene] = [ProblemIntroduction, IndicatorVisualization, CalculationScene]
        for s in scenes:
            # Passing self to the scene's construct method to combine animations
            s.construct(self)
            fade_out(self)
            self.wait(0.5)
