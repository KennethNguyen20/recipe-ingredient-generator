import { bootstrapApplication } from '@angular/platform-browser';
import { RouterModule, Routes } from '@angular/router';
import { AppComponent } from './app/app.component';
import { RecipeComponent } from './app/recipe/recipe.component';
import { importProvidersFrom } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';

// Define your routes
const routes: Routes = [
  { path: 'recipes', component: RecipeComponent }, // Path for RecipeComponent
  { path: '', redirectTo: '/recipes', pathMatch: 'full' }, // Default route redirects to /recipes
];

// Bootstrap the application with the RouterModule and HTTP client
bootstrapApplication(AppComponent, {
  providers: [
    importProvidersFrom(RouterModule.forRoot(routes)), // Import routing
    provideHttpClient(), // Provide the HTTP client
  ],
}).catch((err) => console.error(err));
