import { provideRouter, RouterModule } from '@angular/router';
import { importProvidersFrom } from '@angular/core';
import { bootstrapApplication } from '@angular/platform-browser';
import { AppComponent } from './app.component';
import { RecipeComponent } from './recipe/recipe.component'; // Adjust path as necessary

const routes = [
  { path: '', component: RecipeComponent }, // Default route
];

bootstrapApplication(AppComponent, {
  providers: [provideRouter(routes), importProvidersFrom(RouterModule)],
});
