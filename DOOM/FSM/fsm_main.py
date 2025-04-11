import vizdoom as vzd
import numpy as np
import cv2
import os

class DoomFSM:
    def __init__(self, map_name="basic"):
        self.game = vzd.DoomGame()
        self.map_name = map_name
        self.game.load_config(os.path.join(vzd.scenarios_path, f"{map_name}.cfg"))
        self.game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, f"{map_name}.wad"))
        self.game.set_available_game_variables([vzd.GameVariable.AMMO2, vzd.GameVariable.KILLCOUNT])
        self.game.init()

        # Initialize the attributes
        self.state = "SEARCH"
        self.enemy_centered = False
        self.enemy_detected = False
        
    def get_screen_state(self) -> cv2.typing.MatLike:
        """
        Prepares the screen buffer for processing
        """
        state = self.game.get_state()
        if state:
            screen_buffer = state.screen_buffer  # Raw RGB image
            img = np.transpose(screen_buffer, (1, 2, 0)) # Convert format from (C, H, W) to (H, W, C)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Convert to OpenCV format
            
            return img

    def detect_enemies(self) -> list[cv2.typing.MatLike]:
        """
        Enemy detection using edge detection and contour detection
        """
        # Image processing
        img = self.get_screen_state()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        median_blur_img = cv2.medianBlur(gray_img, 5)
        edges = cv2.Canny(median_blur_img, 100, 200)
        
        # Define the detection region
        top_border = 50
        bottom_border = 115
        left_border = right_border = 30
        h, w = edges.shape

        # Zero out the edges outside the region
        edges[np.r_[0:top_border, h - bottom_border:h], :] = 0
        edges[:, np.r_[0:left_border, w - right_border:w]] = 0
        
        # Find contours (potential enemies)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size to remove noise
        min_contour_area = 5  # Tune based on enemy size
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        
        # Draw bounding boxes around detected enemies on the original full-color image
        enemy_img = img.copy()
        for cnt in filtered_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(enemy_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box

        # Display vision and enemy detection side by side
        combined_img = np.hstack((cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), enemy_img))
        cv2.imshow("Agent's Vision              Enemy Detection", combined_img)
        cv2.waitKey(1)
        
        return filtered_contours
    
    def transition(self) -> None:
        """
        Handles state transitions
        """
        contours = self.detect_enemies()
        self.enemy_detected = True if len(contours) > 0 else False
        
        if self.state == "SEARCH":
            # Enemy detected: switch to AIM state
            if self.enemy_detected:
                self.state = "AIM"
            
        elif self.state == "AIM":
            # No enemies detected: switch to SEARCH state
            if not self.enemy_detected:
                self.state = "SEARCH"
                
            # Enemy in the center of the screen: switch to ATTACK state
            elif self.enemy_centered:
                self.state = "ATTACK"
            
        elif self.state == "ATTACK":
            # No enemies in the center of the screen: switch to SEARCH state
            if not self.enemy_centered:
                self.state = "AIM"
        
    def action(self) -> None:
        """
        Decides the action based on the current state
        """
        if self.state == "SEARCH":
            # Call search method
            action = self.search()
            self.game.make_action(action)
        
        elif self.state == "AIM":
            # Call aim method
            direction = self.aim()
            if direction == "left":
                self.game.make_action([1, 0, 0])
            elif direction == "right":
                self.game.make_action([0, 1, 0])
                
        elif self.state == "ATTACK":
            # Call attack method
            if self.attack() == "attack":
                self.game.make_action([0, 0, 1])
    
    def nearest_enemy(self) -> tuple[int]:
        """
        Finds the nearest enemy based on the horizontal position
        """
        contours = self.detect_enemies()
        if not contours:
            return None
        
        screen = self.get_screen_state()
        h, w, _ = screen.shape
        center_x = w // 2
        
        nearest_contour = None
        min_distance = float("inf")
        
        for cnt in contours:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            enemy_center_x = x + w_box // 2
            distance = abs(enemy_center_x - center_x)
            
            if distance < min_distance:
                min_distance = distance
                nearest_contour = (enemy_center_x, x, w_box, center_x)
        
        return nearest_contour

    def search(self) -> list[int]:
        """
        Returns a left movement action
        """
        return [1, 0, 0]
        
    def aim(self) -> str:
        """
        Adjusts the agent's aim based on the enemy's relative horizontal position
        """
        nearest_contour = self.nearest_enemy()
        if not nearest_contour:
            self.enemy_centered = False
            self.enemy_detected = False
            return
        
        enemy_center_x, x, w_box, center_x = nearest_contour

        tolerance = 2
        enemy_center_x = nearest_contour[0]
        if abs(enemy_center_x - center_x) < tolerance:
            self.enemy_centered = True
            return ""
        else:
            # Adjust turning based on enemy's relative horizontal position
            if enemy_center_x < center_x:
                return "left"
            else:
                return "right"
            
    def attack(self) -> str:
        """
        Attacks the enemy if it is centered
        """
        nearest_contour = self.nearest_enemy()
        if not nearest_contour:
            self.enemy_centered = False
            return ""

        # Unpack values
        enemy_center_x, x, w_box, center_x = nearest_contour
        tolerance = 10
        
        if abs(enemy_center_x - center_x) > tolerance:
            self.enemy_centered = False
            return ""

        return "attack"
        
    def run(self, episodes: int = 5) -> None:
        """
        Runs the game for a specified number of episodes
        """
        for episode in range(episodes):
            self.game.new_episode()
            while not self.game.is_episode_finished():
                self.detect_enemies()  # Perform enemy detection
                self.transition()
                self.action()
            episode_reward = self.game.get_total_reward()
            episode_ticks = self.game.get_episode_time()
            print(f"Episode {episode + 1} Reward: {episode_reward} Time Alive: {episode_ticks}\n")
        self.game.close()





if __name__ == "__main__":
    maps = ["basic", "defend_the_center", "defend_the_line"]
    map_name = maps[1]
    
    doom = DoomFSM(map_name)
    doom.run(1)
