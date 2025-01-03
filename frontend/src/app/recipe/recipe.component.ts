import { Component, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { catchError } from 'rxjs/operators';
import { of } from 'rxjs';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-recipe',
  standalone: true,
  templateUrl: './recipe.component.html',
  styleUrls: ['./recipe.component.css'],
  imports: [CommonModule],
})
export class RecipeComponent {
  private http = inject(HttpClient); // Inject HttpClient
  recipes: any[] = []; // Array to store recommended recipes
  substitutions: { [key: string]: any } = {}; // Object to store substitutions

  // Getter to retrieve the keys of the substitutions object
  get substitutionKeys() {
    return Object.keys(this.substitutions);
  }

  // Function to get recommended recipes based on input ingredients
  getRecipes(ingredients: string) {
    const body = { ingredients };
    this.http
      .post<any>('http://127.0.0.1:5000/recommend', body)
      .pipe(
        catchError((error) => {
          console.error('Error fetching recipes:', error);
          return of({ status: 'error', message: 'Failed to fetch recipes.' });
        })
      )
      .subscribe((response) => {
        if (response.status === 'success') {
          this.recipes = response.data;
        } else {
          console.warn('Recipe response error:', response.message);
        }
      });
  }

  // Function to get ingredient substitutions
  getSubstitutions(ingredients: string) {
    const body = { ingredients };
    this.http
      .post<any>('http://127.0.0.1:5000/substitute', body)
      .pipe(
        catchError((error) => {
          console.error('Error fetching substitutions:', error);
          return of({ status: 'error', message: 'Failed to fetch substitutions.' });
        })
      )
      .subscribe((response) => {
        if (response.status === 'success') {
          this.substitutions = response.data;
        } else {
          console.warn('Substitution response error:', response.message);
        }
      });
  }
}
